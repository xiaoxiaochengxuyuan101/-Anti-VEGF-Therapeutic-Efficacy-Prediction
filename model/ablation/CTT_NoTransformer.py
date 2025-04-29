import cv2  # OpenCV库，用于图像处理
import torch  # PyTorch，用于深度学习建模
import torch.nn as nn  # PyTorch的神经网络模块
import torch.nn.functional as F  # 提供函数式接口的神经网络模块
import torchvision  # PyTorch的计算机视觉库
from einops import rearrange, repeat  # 用于高效的张量操作


class CTT_NoTransformer(nn.Module):
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
        self.hor_feat_fc = nn.Linear(768, self.output_dim)
        self.ver_feat_fc = nn.Linear(768, self.output_dim)
        self.z_encoder = nn.Linear(27, self.output_dim)

        self.fc = nn.Sequential(
            nn.LayerNorm(self.output_dim * 3),
            nn.Linear(self.output_dim * 3, 3)
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

        # 编码结构化文本特征
        text_feature = self.z_encoder(z)

        # 拼接所有特征
        combined_features = torch.cat((hor_feature, ver_feature, text_feature), dim=1)

        # 分类
        output = self.fc(combined_features)
        return output
