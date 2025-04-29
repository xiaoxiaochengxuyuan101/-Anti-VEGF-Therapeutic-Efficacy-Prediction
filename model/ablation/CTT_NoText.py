# import cv2  # OpenCV库，用于图像处理
# import torch  # PyTorch，用于深度学习建模
# import torch.nn as nn  # PyTorch的神经网络模块
# import torch.nn.functional as F  # 提供函数式接口的神经网络模块
# import torchvision  # PyTorch的计算机视觉库
# from einops import rearrange, repeat  # 用于高效的张量操作
#
# class CTT_NoText(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.layers = args.convnext_layers
#         self.output_dim = args.output_dim
#
#         convnext = getattr(torchvision.models, f'convnext_{self.layers}')
#         self.convnext_hor = convnext(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
#         self.convnext_ver = convnext(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
#
#         self.feature_hor = nn.Sequential(*list(self.convnext_hor.features))
#         self.feature_ver = nn.Sequential(*list(self.convnext_ver.features))
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.hor_feat_fc = nn.Linear(768, self.output_dim)
#         self.ver_feat_fc = nn.Linear(768, self.output_dim)
#
#         self.fc = nn.Sequential(
#             nn.LayerNorm(self.output_dim * 2),
#             nn.Linear(self.output_dim * 2, 3)
#         )
#
#     def forward(self, x, y, z=None):
#         # 提取水平方向特征
#         hor_feature = self.feature_hor(x)
#         hor_feature = self.avgpool(hor_feature).flatten(1)
#         hor_feature = self.hor_feat_fc(hor_feature)
#
#         # 提取垂直方向特征
#         ver_feature = self.feature_ver(y)
#         ver_feature = self.avgpool(ver_feature).flatten(1)
#         ver_feature = self.ver_feat_fc(ver_feature)
#
#         # 拼接图像特征
#         combined_features = torch.cat((hor_feature, ver_feature), dim=1)
#
#         # 分类
#         output = self.fc(combined_features)
#         return output

import torch
import torch.nn as nn
import torchvision


# Residual Block
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# PreNorm Block
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # 打印输入形状以便调试
        # print(f"PreNorm input shape: {x.shape}")
        return self.fn(self.norm(x), **kwargs)


# FeedForward Network
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# Attention Block
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).permute(0, 2, 1, 3), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            dots.masked_fill_(~mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)


# Transformer Block
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


# CrossTransformer
class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, nhead, nhid, dropout):
        super().__init__()
        self.transformer_encoder = Transformer(dim, depth, nhead, nhid // nhead, dim * 4, dropout)

    def forward(self, x, z=None, zz=None):  # Ignore z and zz
        p = x  # Only use x
        p = self.transformer_encoder(p)
        x = p[:, 0, :].contiguous().unsqueeze(1)
        return x, None  # Return x and None for z


# Ctt: Cross-transformer topology
class Ctt(nn.Module):
    def __init__(self, dim, nhead, nhid, dropout):
        super().__init__()
        self.Transformer_1 = CrossTransformer(dim, 2, nhead, nhid, dropout)
        self.Transformer_2 = CrossTransformer(dim, 2, nhead, nhid, dropout)
        self.Transformer_x = CrossTransformer(dim, 1, nhead, nhid, dropout)
        self.Transformer_y = CrossTransformer(dim, 1, nhead, nhid, dropout)
        self.layers = 4

    def forward(self, x, y, z1=None, z2=None, z=None):
        x, _ = self.Transformer_1(x, None, None)
        y, _ = self.Transformer_2(y, None, None)

        for layer in range(self.layers):
            x, _ = self.Transformer_x(x, None, None)
            y, _ = self.Transformer_y(y, None, None)

        return None, None, x, y


# MMF: Main Model Framework (with dynamic output_dim)
class CTT_NoText(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = args.convnext_layers
        self.output_dim = args.output_dim

        convnext = getattr(torchvision.models, f'convnext_{self.layers}')
        self.convnext_hor = convnext(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.convnext_ver = convnext(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)

        self.feature_hor = nn.Sequential(*list(self.convnext_hor.features))
        self.feature_ver = nn.Sequential(*list(self.convnext_ver.features))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.Transformer_layer = Ctt(dim=self.output_dim // 2, nhead=4, nhid=1024, dropout=0.1)

        # 根据拼接的维度动态调整 LayerNorm 和全连接层
        self.fc = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.output_dim)
        )

        # 动态调整特征提取层
        self.hor_feat_fc = nn.Linear(768, self.output_dim // 2)
        self.ver_feat_fc = nn.Linear(768, self.output_dim // 2)

    def forward(self, x, y, z=None):  # z is ignored
        hor_feature = self.feature_hor(x)
        hor_feature = self.avgpool(hor_feature)
        hor_feature = torch.flatten(hor_feature, 1).unsqueeze(1)
        hor_feature = self.hor_feat_fc(hor_feature)

        ver_feature = self.feature_ver(y)
        ver_feature = self.avgpool(ver_feature)
        ver_feature = torch.flatten(ver_feature, 1).unsqueeze(1)
        ver_feature = self.ver_feat_fc(ver_feature)

        _, _, hor_feature_n, ver_feature_n = self.Transformer_layer(hor_feature, ver_feature, None, None, None)

        hor_feature_n = hor_feature_n.squeeze(1)
        ver_feature_n = ver_feature_n.squeeze(1)

        combined_features = torch.cat((hor_feature_n, ver_feature_n), dim=1)

        # 分类输出
        class_output = self.fc(combined_features)

        return class_output
