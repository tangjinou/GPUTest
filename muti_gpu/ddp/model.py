"""
模型定义文件
包含神经网络模型的定义
"""

import torch.nn as nn


class SimpleNet(nn.Module):
    """简单的神经网络模型示例"""
    def __init__(self, in_features=10, hidden_size=128, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

