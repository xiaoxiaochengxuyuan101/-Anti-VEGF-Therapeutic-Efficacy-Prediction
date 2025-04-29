import torch
import torch.nn as nn
import torchvision.models as models

class ResNetWithTransformer(nn.Module):
    def __init__(self, num_classes, text_feature_size):
        super(ResNetWithTransformer, self).__init__()
        # CNN 模块：使用 ResNet-18 从 OCT 图像中提取特征
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Identity()  # 移除最终的全连接层，输出大小为 (batch_size, 512)

        # 编码模块
        self.image_fc = nn.Linear(512, 128)  # 将每个图像特征 (512) 转换为 128 维的 token
        # 文本特征编码
        self.text_fc = nn.Sequential(
            nn.Linear(text_feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Transformer 模块
        transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=6)

        # Head 模块用于预测类别（num_classes 类）
        self.head = nn.Linear(128, num_classes)  # 预测类别数量
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, h_image, v_image, text_data):
        # 将水平和垂直图像通过 CNN
        h_features = self.cnn(h_image)  # (batch_size, 512)
        v_features = self.cnn(v_image)  # (batch_size, 512)

        # 将水平和垂直特征通过全连接层处理
        h_features = self.image_fc(h_features)  # (batch_size, 128)
        v_features = self.image_fc(v_features)  # (batch_size, 128)

        # 将水平和垂直特征平均融合
        # combined_features = (h_features + v_features) / 2  # (batch_size, 128)
        combined_features = torch.cat((h_features, v_features), dim=1)
        # 使用全连接层处理文本特征
        text_token = self.text_fc(text_data)  # (batch_size, 128)

        # 将图像特征和文本 token 拼接成一个序列
        # Transformer 的最终输入包括图像特征和文本数据作为独立的 token
        token_sequence = torch.stack((combined_features, text_token), dim=1)  # (batch_size, 2, 128)

        # 通过 Transformer 模块
        transformer_output = self.transformer(token_sequence)  # (batch_size, 2, 128)
        prediction_token = transformer_output[:, 0, :]  # 提取第一个 token 作为最终预测的输入

        # 通过 Head 模块得到预测的类别
        output = self.head(prediction_token)  # (batch_size, num_classes)

        return output
