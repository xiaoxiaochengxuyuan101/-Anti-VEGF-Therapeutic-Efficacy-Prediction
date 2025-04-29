import torch
import torch.nn as nn
import timm

class MultimodalFusionNet(nn.Module):
    def __init__(self, num_classes, text_feature_size):
        super(MultimodalFusionNet, self).__init__()

        # 使用timm加载预训练的Inception-ResNet-v2模型
        self.cnn = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=0)  # num_classes=0 不输出分类结果
        self.cnn.fc = nn.Identity()  # 移除原本的全连接层，只输出特征向量
        self.cnn_output_dim = 1536  # Inception-ResNet-v2的特征输出维度（1536）

        # 添加降维模块：将1536维压缩到128维
        self.cnn_dim_reduction = nn.Sequential(
            nn.Linear(self.cnn_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5)  # 防止过拟合
        )

        # MLP用于处理结构化数据
        self.mlp = nn.Sequential(
            nn.Linear(text_feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 融合CNN和MLP特征
        self.fc1 = nn.Linear(128 + 128 + 64, 128)  # h_image + v_image + mlp
        self.fc2 = nn.Linear(128, num_classes)  # 最终分类或回归输出

    def forward(self, h_image, v_image, text_data):
        # 提取水平图像特征
        h_features = self.cnn(h_image)  # 返回特征向量
        h_features = self.cnn_dim_reduction(h_features)  # 压缩为128维

        # 提取垂直图像特征
        v_features = self.cnn(v_image)  # 返回特征向量
        v_features = self.cnn_dim_reduction(v_features)  # 压缩为128维

        # 提取结构化数据特征
        mlp_features = self.mlp(text_data)  # 输出维度: (batch_size, 64)

        # 融合所有特征
        combined = torch.cat((h_features, v_features, mlp_features), dim=1)

        # 全连接层
        x = torch.relu(self.fc1(combined))
        output = self.fc2(x)  # 最终输出: (batch_size, num_classes)
        return output
