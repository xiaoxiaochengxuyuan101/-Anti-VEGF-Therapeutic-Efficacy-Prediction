# import torch
# import torch.nn as nn
# import torchvision
#
# class Transformer(nn.Module):
#     """单模态的标准Transformer模块"""
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 nn.LayerNorm(dim),
#                 nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout),
#                 nn.LayerNorm(dim),
#                 nn.Sequential(
#                     nn.Linear(dim, mlp_dim),
#                     nn.GELU(),
#                     nn.Linear(mlp_dim, dim),
#                     nn.Dropout(dropout)
#                 )
#             ]))
#
#     def forward(self, x):
#         for norm1, attn, norm2, ff in self.layers:
#             # Self-Attention
#             x_norm = norm1(x)
#             x, _ = attn(x_norm, x_norm, x_norm)
#             # Feed Forward
#             x = x + ff(norm2(x))
#         return x
#
# class CTT_NoCrossInteraction(nn.Module):
#     """去除跨模态交互，仅保留单模态Transformer的模型"""
#     def __init__(self, args):
#         super().__init__()
#         self.layers = args.convnext_layers
#         self.output_dim = args.output_dim
#
#         # ConvNeXt 模块
#         convnext = getattr(torchvision.models, f'convnext_{self.layers}')
#         self.convnext_hor = convnext(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
#         self.convnext_ver = convnext(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
#
#         self.feature_hor = nn.Sequential(*list(self.convnext_hor.features))
#         self.feature_ver = nn.Sequential(*list(self.convnext_ver.features))
#
#         # Transformer 模块（单模态处理）
#         self.hor_transformer = Transformer(dim=self.output_dim, depth=2, heads=4, dim_head=64, mlp_dim=1024)
#         self.ver_transformer = Transformer(dim=self.output_dim, depth=2, heads=4, dim_head=64, mlp_dim=1024)
#         self.text_transformer = Transformer(dim=self.output_dim, depth=2, heads=4, dim_head=64, mlp_dim=1024)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#         # 映射到Transformer输入维度
#         self.hor_proj = nn.Linear(768, self.output_dim)
#         self.ver_proj = nn.Linear(768, self.output_dim)
#         self.text_proj = nn.Linear(27, self.output_dim)
#
#         # 分类头
#         self.fc = nn.Sequential(
#             nn.LayerNorm(self.output_dim * 3),
#             nn.Linear(self.output_dim * 3, 3)
#         )
#
#     def forward(self, x, y, z):
#         # 水平图像特征
#         hor_feature = self.feature_hor(x)
#         hor_feature = self.avgpool(hor_feature).flatten(1)
#         hor_feature = self.hor_proj(hor_feature).unsqueeze(1)  # (B, 1, output_dim)
#
#         # 垂直图像特征
#         ver_feature = self.feature_ver(y)
#         ver_feature = self.avgpool(ver_feature).flatten(1)
#         ver_feature = self.ver_proj(ver_feature).unsqueeze(1)  # (B, 1, output_dim)
#
#         # 文本特征
#         text_feature = self.text_proj(z).unsqueeze(1)  # (B, 1, output_dim)
#
#         # 分别通过各自的Transformer
#         hor_feature = self.hor_transformer(hor_feature)  # (B, 1, output_dim)
#         ver_feature = self.ver_transformer(ver_feature)  # (B, 1, output_dim)
#         text_feature = self.text_transformer(text_feature)  # (B, 1, output_dim)
#
#         # 去除多余维度
#         hor_feature = hor_feature.squeeze(1)
#         ver_feature = ver_feature.squeeze(1)
#         text_feature = text_feature.squeeze(1)
#
#         # 特征拼接
#         combined_features = torch.cat((hor_feature, ver_feature, text_feature), dim=1)
#
#         # 分类
#         output = self.fc(combined_features)
#         return output


import torch
import torch.nn as nn
import torchvision

class CTT_NoInteraction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = args.convnext_layers  # ConvNeXt 层级，例如 'tiny'
        self.output_dim = args.output_dim

        # 加载 ConvNeXt 模型
        convnext = getattr(torchvision.models, f'convnext_{self.layers}')
        self.convnext_hor = convnext(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.convnext_ver = convnext(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)

        self.feature_hor = nn.Sequential(*list(self.convnext_hor.features))  # 提取水平特征
        self.feature_ver = nn.Sequential(*list(self.convnext_ver.features))  # 提取垂直特征

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化
        self.hor_feat_fc = nn.Linear(768, self.output_dim)  # ConvNeXt 输出到目标维度
        self.ver_feat_fc = nn.Linear(768, self.output_dim)

        # Transformer 编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.output_dim, nhead=4, dim_feedforward=1024, dropout=args.dropout),
            num_layers=2
        )

        # 结构化文本特征
        self.z_encoder = nn.Linear(27, self.output_dim)

        # 最终分类层
        self.fc = nn.Sequential(
            nn.LayerNorm(self.output_dim * 3),  # 结合水平、垂直和文本特征
            nn.Linear(self.output_dim * 3, 3)  # 三分类输出
        )

    def forward(self, x, y, z):
        # 提取水平方向特征
        hor_feature = self.feature_hor(x)
        hor_feature = self.avgpool(hor_feature).flatten(1)
        hor_feature = self.hor_feat_fc(hor_feature)

        # 提取垂直方向特征
        ver_feature = self.feature_ver(y)
        ver_feature = self.avgpool(ver_feature).flatten(1)
        ver_feature = self.ver_feat_fc(ver_feature)

        # 编码文本特征
        text_feature = self.z_encoder(z)

        # 水平和垂直特征分别通过 Transformer（无交互）
        hor_feature = self.transformer(hor_feature.unsqueeze(1)).squeeze(1)  # Transformer 输入维度：[B, N, C]
        ver_feature = self.transformer(ver_feature.unsqueeze(1)).squeeze(1)

        # 拼接三个特征
        combined_features = torch.cat((hor_feature, ver_feature, text_feature), dim=1)

        # 分类
        output = self.fc(combined_features)
        return output
