import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat

from collections import OrderedDict

class Resnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dp = args.dropout
        self.layers = args.resnet_layers
        self.output_dim = args.output_dim
        model = getattr(torchvision.models, 'resnet{}'.format(self.layers))
        self.resnet = model(pretrained=True)
        self.feature = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4)

        self.fc = nn.Linear(512, self.output_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
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
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
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

class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, nhead, nhid, dropout):
        super().__init__()
        self.transformer_encoder = Transformer(dim, depth, nhead, nhid // nhead, dim * 4, dropout)

    def forward(self, x, z, zz):
        p = torch.cat((x, z, zz), 1)
        p = self.transformer_encoder(p)
        x = p[:, 0, :].contiguous().unsqueeze(1)
        z = p[:, 1, :].contiguous().unsqueeze(1)
        return x, z

class Ctt(nn.Module):
    def __init__(self, dim, nhead, nhid, dropout):
        super().__init__()
        self.Transformer_1 = CrossTransformer(dim, 2, nhead, nhid, dropout)
        self.Transformer_2 = CrossTransformer(dim, 2, nhead, nhid, dropout)
        self.Transformer_x = CrossTransformer(dim, 1, nhead, nhid, dropout)
        self.Transformer_y = CrossTransformer(dim, 1, nhead, nhid, dropout)
        self.layers = 4

    def forward(self, x, y, z1, z2, z):
        x, z1 = self.Transformer_1(x, z1, z)
        y, z2 = self.Transformer_2(y, z2, z)

        for layer in range(self.layers):
            zt = z1
            z1 = z2
            z2 = zt
            x, z1 = self.Transformer_x(x, z1, z)
            y, z2 = self.Transformer_y(y, z2, z)

        return z1, z2, x, y

class MMF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = args.resnet_layers
        self.output_dim = args.output_dim

        resnet = getattr(torchvision.models, 'resnet{}'.format(self.layers))
        self.resnet_hor = resnet(pretrained=True)
        self.resnet_ver = resnet(pretrained=True)

        self.feature_hor = nn.Sequential(
            self.resnet_hor.conv1,
            self.resnet_hor.bn1,
            self.resnet_hor.relu,
            self.resnet_hor.maxpool,
            self.resnet_hor.layer1,
            self.resnet_hor.layer2,
            self.resnet_hor.layer3,
            self.resnet_hor.layer4)
        self.feature_ver = nn.Sequential(
            self.resnet_ver.conv1,
            self.resnet_ver.bn1,
            self.resnet_ver.relu,
            self.resnet_ver.maxpool,
            self.resnet_ver.layer1,
            self.resnet_ver.layer2,
            self.resnet_ver.layer3,
            self.resnet_ver.layer4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.z_hor_encoder = nn.Linear(27, self.output_dim)  # 输入特征数为27
        self.z_ver_encoder = nn.Linear(27, self.output_dim)  # 输入特征数为27，输出特征数为128
        self.z_cva_encoder = nn.Linear(27, self.output_dim)

        self.hor_reg_token = nn.Parameter(torch.randn(1, 1))
        self.ver_reg_token = nn.Parameter(torch.randn(1, 1))

        self.Transformer_layer = Ctt(dim=self.output_dim, nhead=4, nhid=1024, dropout=0.1)

        self.fc = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, self.output_dim)
        )

        self.cva_fc = nn.Linear(256, 3)  # 修改为三分类
        self.hor_feat_fc = nn.Linear(2048, self.output_dim)  # 使用2048作为输入特征数量
        self.ver_feat_fc = nn.Linear(2048, self.output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.top = nn.Linear(self.output_dim, 3)  # 修改为三分类

    def forward(self, x, y, z):
        hor_feature = x
        ver_feature = y

        hor_feature = self.feature_hor(hor_feature)
        hor_feature = self.resnet_hor.avgpool(hor_feature)
        hor_feature = torch.flatten(hor_feature, 1).unsqueeze(1)
        hor_feature = self.hor_feat_fc(hor_feature)

        ver_feature = self.feature_ver(ver_feature)
        ver_feature = self.resnet_ver.avgpool(ver_feature)
        ver_feature = torch.flatten(ver_feature, 1).unsqueeze(1)
        ver_feature = self.ver_feat_fc(ver_feature)

        hor_cva = self.z_hor_encoder(z).unsqueeze(dim=1)
        ver_cva = self.z_ver_encoder(z).unsqueeze(dim=1)
        cva_token = self.z_cva_encoder(z).unsqueeze(dim=1)

        hor_cva, ver_cva, hor_feature_n, ver_feature_n = self.Transformer_layer(hor_feature, ver_feature, hor_cva,
                                                                                 ver_cva, cva_token)

        hor_cva = hor_cva.squeeze(1)
        ver_cva = ver_cva.squeeze(1)
        hor_feature = hor_feature.squeeze(1)
        ver_feature = ver_feature.squeeze(1)
        hor_feature_n = hor_feature_n.squeeze(1)
        ver_feature_n = ver_feature_n.squeeze(1)

        # Combine features for classification
        combined_features = torch.cat((hor_cva, ver_cva, hor_feature_n, ver_feature_n), dim=1)

        # Final classification
        class_output = self.fc(combined_features)

        return class_output

    def model_name(self):
        return 'MMF'

