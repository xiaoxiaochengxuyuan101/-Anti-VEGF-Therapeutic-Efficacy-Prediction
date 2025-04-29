import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm  # 用于显示训练进度条

from dataset.dataset import MyDataSet

from torchvision.models import ResNet50_Weights


class ResNetWithText(nn.Module):
    def __init__(self, num_classes, text_feature_size):
        super(ResNetWithText, self).__init__()
        # 使用最新的方式加载预训练的ResNet50模型
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features

        # 替换ResNet的最后一层全连接层
        self.resnet.fc = nn.Identity()  # 先移除最后的全连接层

        # 文本特征分支
        self.text_fc = nn.Sequential(
            nn.Linear(text_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 最后的全连接层
        self.fc = nn.Linear(num_ftrs + 512, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, h_image, v_image, text_data):
        # 将水平和垂直图像输入ResNet模型
        h_features = self.resnet(h_image)
        v_features = self.resnet(v_image)

        # 合并图像特征
        image_features = h_features + v_features

        # 文本特征处理
        text_features = self.text_fc(text_data)

        # 合并图像特征和文本特征
        combined_features = torch.cat((image_features, text_features), dim=1)

        # 输出最终的分类结果
        output = self.fc(combined_features)
        return output

