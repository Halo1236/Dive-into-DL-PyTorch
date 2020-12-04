#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
created by Halo 2020/11/19 16:52
"""

import torch
from torch import nn

"""
池化（pooling）层，它的提出是为了缓解卷积层对位置的过度敏感性。
减少了参数数量，从而可以预防网络过拟合

实施池化的目的：(1) 降低信息冗余；(2) 提升模型的尺度不变性、旋转不变性；(3) 防止过拟合。

池化层的常见操作包含以下几种：最大值池化，均值池化，随机池化，中值池化，组合池化等。

"""


def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
Y = pool2d(X, (2, 2), 'avg')
print(Y)

"""
我们将通过nn模块里的二维最大池化层MaxPool2d来演示池化层填充和步幅的工作机制。我们先构造一个形状为(1, 1, 4, 4)的输入数据，前两个维度分别是批量和通道
"""

X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
print(X)
pl2d = nn.MaxPool2d(3)
print(pl2d(X))
pool2d = nn.MaxPool2d(3, padding=1, stride=3)
print(pool2d(X))
pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
print(pool2d(X))

"""
池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入按通道相加。
"""
X = torch.cat((X, X + 1), dim=1)
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))