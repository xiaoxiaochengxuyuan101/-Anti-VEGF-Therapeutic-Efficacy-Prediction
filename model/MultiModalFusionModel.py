import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat

# ResNet-based feature extractor
class ResnetFeatureExtractor(nn.Module):
    def __init__(self, resnet_layers=18, output_dim=128, dropout=0.1):
        super().__init__()
        model = getattr(torchvision.models, f'resnet{resnet_layers}')(pretrained=True)
        self.feature_extractor = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4
        )
        self.fc = nn.Linear(512, output_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(self.dropout(x))
        return x

# Transformer-based cross-modal fusion module
class CrossTransformer(nn.Module):
    def __init__(self, dim, depth=3, heads=4, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.transformer = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ])
            for _ in range(depth)
        ])

    def forward(self, x, z, zz):
        combined = torch.cat((x, z, zz), dim=1)
        for attn, ff in self.transformer:
            combined = attn(combined)
            combined = ff(combined)
        return combined[:, 0:1, :], combined[:, 1:2, :]

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

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

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        inner_dim = heads * (dim // heads)
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# The complete multi-modal fusion model
class MultiModalFusionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        resnet_output_dim = 128  # Output dim for ResNet
        self.resnet_horizontal = ResnetFeatureExtractor(args.resnet_layers, resnet_output_dim, args.dropout)
        self.resnet_vertical = ResnetFeatureExtractor(args.resnet_layers, resnet_output_dim, args.dropout)

        # Text feature encoders
        self.z_horizontal_encoder = nn.Linear(27, 128)  # Adjust for your text feature size
        self.z_vertical_encoder = nn.Linear(27, 128)
        self.z_cva_encoder = nn.Linear(27, 128)

        # Cross-modal Transformer layer
        self.transformer_layer = CrossTransformer(dim=128, depth=3, heads=4, mlp_dim=512, dropout=args.dropout)

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, args.num_classes)
        )

    def forward(self, horizontal_image, vertical_image, additional_features):
        horizontal_feature = self.resnet_horizontal(horizontal_image)
        vertical_feature = self.resnet_vertical(vertical_image)

        # Encode the additional features
        horizontal_cva = self.z_horizontal_encoder(additional_features).unsqueeze(1)
        vertical_cva = self.z_vertical_encoder(additional_features).unsqueeze(1)
        cva_token = self.z_cva_encoder(additional_features).unsqueeze(1)

        # Cross-token fusion using transformer
        horizontal_cva, vertical_cva = self.transformer_layer(
            horizontal_feature.unsqueeze(1),
            vertical_feature.unsqueeze(1),
            cva_token
        )

        # Combine features for final classification
        combined_features = torch.cat((horizontal_cva.squeeze(1), vertical_cva.squeeze(1)), dim=1)
        class_output = self.fc(combined_features)
        return class_output

    def model_name(self):
        return "MultiModalFusionModel"

