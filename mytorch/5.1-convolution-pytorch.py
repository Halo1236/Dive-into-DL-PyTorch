#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
created by Halo 2020/11/16 15:32
"""

import torch
from torch import nn


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)
K = torch.tensor([[1, -1]])

Y = corr2d(X, K)
print(Y)

"""
最后我们来看一个例子，它使用物体边缘检测中的输入数据X和输出数据Y来学习我们构造的核数组K。我们首先构造一个卷积层，其卷积核将被初始化成随机数组。接下来在每一次迭代中，我们使用平方误差来比较Y和卷积层的输出，然后计算梯度来更新权重。
"""
# 构造一个核数组形状是(1, 2)的二维卷积层

conv2d = Conv2D(kernel_size=(1, 2))

step = 50
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad
    """
    ①对于修改tensor的数值，但是又不希望被autograd记录（即不会影响反向传播），那么我么可以对tensor.data进行操作。

    ②对于被torch.no_grad():包裹的tensor，在反向传播的过程中，其梯度是不会回传的
    可修改为：
    with torch.no_grad(﻿)﻿:

        conv2d.weight -= lr * conv2d.weight.grad

        conv2d.bias -= lr * conv2d.bias.grad
    """

    # 梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.zero_()
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))
print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)
