import cv2  # OpenCV库，用于图像处理
import torch  # PyTorch，用于深度学习建模
import torch.nn as nn  # PyTorch的神经网络模块
import torch.nn.functional as F  # 提供函数式接口的神经网络模块
import torchvision  # PyTorch的计算机视觉库
from einops import rearrange, repeat  # 用于高效的张量操作

from collections import OrderedDict  # 提供有序字典的数据结构

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # 保存传入的函数模块

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x  # 执行函数并添加输入，实现残差连接

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 初始化层归一化模块
        self.fn = fn  # 保存传入的函数模块

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # 在执行函数前对输入进行归一化

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 第一层线性变换
            nn.GELU(),  # GELU激活函数
            nn.Dropout(dropout),  # Dropout正则化
            nn.Linear(hidden_dim, dim),  # 第二层线性变换
            nn.Dropout(dropout)  # Dropout正则化
        )

    def forward(self, x):
        return self.net(x)  # 执行前馈网络计算

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 内部维度 = 头数 * 每个头的维度
        self.heads = heads  # 注意力头数
        self.scale = dim ** -0.5  # 缩放因子，用于点积计算
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 生成qkv的线性变换
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 输出映射回原始维度
            nn.Dropout(dropout)  # Dropout正则化
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads  # 获取输入维度信息和头数
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 生成q, k, v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # 重排张量维度
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # 点积计算注意力权重
        mask_value = -torch.finfo(dots.dtype).max  # 无效掩码值

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)  # 填充掩码
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]  # 广播掩码
            dots.masked_fill_(~mask, mask_value)  # 填充无效位置

        attn = dots.softmax(dim=-1)  # 计算softmax注意力
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # 通过注意力权重加权v
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重排输出张量
        out = self.to_out(out)  # 映射回原始维度
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])  # 初始化空模块列表
        for _ in range(depth):  # 按深度堆叠Transformer层
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),  # 多头注意力
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))  # 前馈网络
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:  # 遍历每层
            x = attn(x, mask=mask)  # 执行注意力机制
            x = ff(x)  # 执行前馈网络
        return x

class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, nhead, nhid, dropout):
        super().__init__()
        self.transformer_encoder = Transformer(dim, depth, nhead, nhid // nhead, dim * 4, dropout)  # Transformer编码器

    def forward(self, x, z, zz):
        p = torch.cat((x, z, zz), 1)  # 拼接输入张量
        p = self.transformer_encoder(p)  # 通过Transformer编码
        x = p[:, 0, :].contiguous().unsqueeze(1)  # 提取x特征
        z = p[:, 1, :].contiguous().unsqueeze(1)  # 提取z特征
        return x, z

class Ctt(nn.Module):
    def __init__(self, dim, nhead, nhid, dropout):
        super().__init__()
        self.Transformer_1 = CrossTransformer(dim, 2, nhead, nhid, dropout)  # 第一个交叉Transformer
        self.Transformer_2 = CrossTransformer(dim, 2, nhead, nhid, dropout)  # 第二个交叉Transformer
        self.Transformer_x = CrossTransformer(dim, 1, nhead, nhid, dropout)  # x方向的交叉Transformer
        self.Transformer_y = CrossTransformer(dim, 1, nhead, nhid, dropout)  # y方向的交叉Transformer
        self.layers = 4  # 交替层数

    def forward(self, x, y, z1, z2, z):
        x, z1 = self.Transformer_1(x, z1, z)  # 第一轮Transformer操作
        y, z2 = self.Transformer_2(y, z2, z)  # 第二轮Transformer操作

        for layer in range(self.layers):  # 多轮交替处理
            zt = z1  # 交换z1和z2
            z1 = z2
            z2 = zt
            x, z1 = self.Transformer_x(x, z1, z)  # x方向更新
            y, z2 = self.Transformer_y(y, z2, z)  # y方向更新

        return z1, z2, x, y

class MMF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = args.convnext_layers  # ConvNeXt层数
        self.output_dim = args.output_dim  # 输出维度

        convnext = getattr(torchvision.models, f'convnext_{self.layers}')  # 获取指定的ConvNeXt模型
        self.convnext_hor = convnext(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.convnext_ver = convnext(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)

        self.feature_hor = nn.Sequential(*list(self.convnext_hor.features))  # 水平特征提取层
        self.feature_ver = nn.Sequential(*list(self.convnext_ver.features))  # 垂直特征提取层

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化
        self.z_hor_encoder = nn.Linear(27, self.output_dim)  # 水平z特征编码
        self.z_ver_encoder = nn.Linear(27, self.output_dim)  # 垂直z特征编码
        self.z_cva_encoder = nn.Linear(27, self.output_dim)  # z融合特征编码

        self.hor_reg_token = nn.Parameter(torch.randn(1, 1))  # 水平方向的注册令牌
        self.ver_reg_token = nn.Parameter(torch.randn(1, 1))  # 垂直方向的注册令牌

        self.Transformer_layer = Ctt(dim=self.output_dim, nhead=4, nhid=1024, dropout=0.1)  # CTT跨模态模块

        self.fc = nn.Sequential(
            nn.LayerNorm(1024),  # 层归一化
            nn.Linear(1024, self.output_dim)  # 最后一层全连接
        )

        self.cva_fc = nn.Linear(256, 3)  # 分类任务头
        self.hor_feat_fc = nn.Linear(768, self.output_dim)  # 水平特征映射
        self.ver_feat_fc = nn.Linear(768, self.output_dim)  # 垂直特征映射

        self.relu = nn.ReLU()  # 激活函数
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活

        self.top = nn.Linear(self.output_dim, 3)  # 最终输出头

    def forward(self, x, y, z):
        hor_feature = x  # 水平输入
        ver_feature = y  # 垂直输入

        hor_feature = self.feature_hor(hor_feature)  # 提取水平特征
        hor_feature = self.avgpool(hor_feature)  # 平均池化
        hor_feature = torch.flatten(hor_feature, 1).unsqueeze(1)  # 展平并增加维度
        hor_feature = self.hor_feat_fc(hor_feature)  # 映射到目标维度

        ver_feature = self.feature_ver(ver_feature)  # 提取垂直特征
        ver_feature = self.avgpool(ver_feature)  # 平均池化
        ver_feature = torch.flatten(ver_feature, 1).unsqueeze(1)  # 展平并增加维度
        ver_feature = self.ver_feat_fc(ver_feature)  # 映射到目标维度

        hor_cva = self.z_hor_encoder(z).unsqueeze(dim=1)  # 编码水平z特征
        ver_cva = self.z_ver_encoder(z).unsqueeze(dim=1)  # 编码垂直z特征
        cva_token = self.z_cva_encoder(z).unsqueeze(dim=1)  # 编码融合z特征

        hor_cva, ver_cva, hor_feature_n, ver_feature_n = self.Transformer_layer(hor_feature, ver_feature, hor_cva,
                                                                                 ver_cva, cva_token)  # 跨模态融合

        hor_cva = hor_cva.squeeze(1)  # 去除多余维度
        ver_cva = ver_cva.squeeze(1)  # 去除多余维度
        hor_feature = hor_feature.squeeze(1)  # 去除多余维度
        ver_feature = ver_feature.squeeze(1)  # 去除多余维度
        hor_feature_n = hor_feature_n.squeeze(1)  # 去除多余维度
        ver_feature_n = ver_feature_n.squeeze(1)  # 去除多余维度

        # Combine features for classification
        combined_features = torch.cat((hor_cva, ver_cva, hor_feature_n, ver_feature_n), dim=1)  # 拼接特征

        # Final classification
        class_output = self.fc(combined_features)  # 最终分类输出

        return class_output

    def model_name(self):
        return 'MMF'  # 返回模型名称
