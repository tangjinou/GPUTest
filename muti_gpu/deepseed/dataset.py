"""
数据集定义文件
包含数据集类的定义
"""

import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """示例数据集 - 创建有真实关系的数据"""
    def __init__(self, size=10000, features=10, noise_level=0.1):
        """
        Args:
            size: 数据集大小
            features: 特征维度
            noise_level: 噪声水平（0-1之间，越小越容易学习）
        """
        # 生成随机特征
        self.x = torch.randn(size, features)
        
        # 创建一个真实的分类规则：基于特征的加权和
        # 使用前5个特征创建一个可学习的模式
        weights = torch.tensor([1.0, -0.5, 0.8, -0.3, 0.6] + [0.0] * (features - 5))
        
        # 计算加权和并添加噪声
        logits = (self.x * weights).sum(dim=1) + torch.randn(size) * noise_level
        
        # 转换为二分类标签（使用sigmoid阈值）
        self.y = (torch.sigmoid(logits) > 0.5).long()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

