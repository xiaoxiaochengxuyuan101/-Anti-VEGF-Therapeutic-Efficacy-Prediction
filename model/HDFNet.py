import torch
import torch.nn as nn
import torch.nn.functional as F

class HDFNet(nn.Module):
    def __init__(self, num_classes, text_feature_size):
        super(HDFNet, self).__init__()

        # 特征提取网络（用于水平和垂直图像）
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=1)  # Conv1, 11x11 conv, 96
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1)  # Conv2, 5x5 conv, 256
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)  # Conv3, 3x3 conv, 384
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)  # Conv4, 3x3 conv, 384
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)  # Conv5, 3x3 conv, 256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 动态计算展平后的特征图大小
        self.flattened_size = self._get_flattened_size()

        # 图像特征全连接层
        self.fc_image = nn.Linear(self.flattened_size, 4096)
        self.batch_norm_image = nn.BatchNorm1d(4096)

        # 数值特征输入层
        self.fc_numeric = nn.Linear(text_feature_size, 256)  # 将数值特征映射到256维

        # 融合特征全连接层
        self.fc_combined = nn.Linear(4096 * 2 + 256, 4096)  # 融合特征的输入大小为 4096 + 4096 + 256 = 8704
        self.fc_out = nn.Linear(4096, num_classes)  # 最终分类输出

    def _get_flattened_size(self):
        # 使用一个虚拟的输入计算展平后的特征图大小
        x = torch.zeros(1, 3, 224, 224)  # 假设输入图像大小为224x224
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        return x.view(1, -1).size(1)

    def feature_extractor(self, x_image):
        # 图像特征提取网络
        x = F.relu(self.conv1(x_image))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        # 展平
        x = x.view(x.size(0), -1)  # 展平为二维张量
        x = F.relu(self.fc_image(x))
        x = self.batch_norm_image(x)
        return x

    def forward(self, h_image, v_image, x_numeric):
        # 通过特征提取网络获取水平图像的特征
        h_feature = self.feature_extractor(h_image)

        # 通过特征提取网络获取垂直图像的特征
        v_feature = self.feature_extractor(v_image)

        # 数值特征编码
        x_numeric = F.relu(self.fc_numeric(x_numeric))

        # 融合水平图像特征、垂直图像特征和数值特征
        combined_feature = torch.cat((h_feature, v_feature, x_numeric), dim=1)

        # 分类网络
        x = F.relu(self.fc_combined(combined_feature))
        x = self.fc_out(x)
        return x
