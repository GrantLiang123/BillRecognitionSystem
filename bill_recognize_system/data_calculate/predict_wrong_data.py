"""
该模块实现了基于神经网络的错误数据识别方法。通过生成包含正常数据和异常值
的二维数据集，并对数据进行标准化处理。然后，构建了一个多层感知器（MLP）神经网络模型，
包括多个隐藏层和输出层，用于对数据进行分类，判断数据点是否为异常值。
模型训练采用二元交叉熵损失函数和Adam优化器，在训练过程中通过学习率调度器
动态调整学习率。最后（测试中使用），评估模型的性能并可视化识别结果，以实现对数据中错误值的有效识别和检测。
"""

# coding : utf-8
# Time : 2024/6/3
# Author : Liang Zi Fan
# File : regression_prediction.py
# Software : Pycharm

import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def prepare_data_test():
    # 数据准备
    train_data_length = 1024
    train_data = torch.zeros((train_data_length, 2))
    train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)  # x
    train_data[:, 1] = torch.sin(train_data[:, 0])  # y

    # 添加异常值
    num_outliers = 50
    outliers = torch.zeros((num_outliers, 2))
    outliers[:, 0] = 2 * math.pi * torch.rand(num_outliers)
    outliers[:, 1] = 2 * (torch.rand(num_outliers) - 0.5)  # 随机噪声作为异常值

    # 合并正常数据和异常值
    train_data = torch.cat([train_data, outliers], dim=0)
    train_labels = torch.cat([torch.zeros(train_data_length), torch.ones(num_outliers)], dim=0)  # 0表示正常数据，1表示异常值

    # 添加标志位，标记手动添加的异常值
    flag_normal = torch.zeros(train_data_length, 1)
    flag_outliers = torch.ones(num_outliers, 1)
    flags = torch.cat([flag_normal, flag_outliers], dim=0)

    # 数据标准化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)

    # 创建数据集和数据加载器
    dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return train_data, train_labels, train_loader, flags


# 定义神经网络模型
class ImprovedNN(nn.Module):
    def __init__(self):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x


def build_model(train_data, train_labels, train_loader, flags):
    model = ImprovedNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 训练模型
    num_epochs = 200  # 训练次数

    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        scheduler.step()  # 更新学习率

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 评估模型性能
    model.eval()
    with torch.no_grad():
        predictions = model(train_data).squeeze()
        predicted_labels = (predictions >= 0.15).float()
        accuracy = (predicted_labels == train_labels).sum().item() / train_labels.size(0)
        print(f'Accuracy: {accuracy:.4f}')

    # 找出被模型标记为错误的数据点
    outliers_detected = train_data[predicted_labels == 1]
    detected_flags = flags[predicted_labels == 1]
    print(outliers_detected)
    print(len(outliers_detected))
    output = outliers_detected[detected_flags.squeeze() == 0]
    print(output)
    print(len(output))

    # 可视化检测结果
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap='coolwarm', label='Data')
    plt.scatter(outliers_detected[:, 0], outliers_detected[:, 1], edgecolor='k', facecolors='none',
                label='Detected Outliers')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return output


if __name__ == '__main__':
    my_train_data, my_train_labels, my_train_loader, my_flags = prepare_data_test()
    build_model(my_train_data, my_train_labels, my_train_loader, my_flags)
