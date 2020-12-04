#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
created by Halo 2020/11/17 15:34
"""

import torch
from torch import nn


# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道


# https://www.jianshu.com/p/45a26d278473 PyTorch中的nn.Conv1d与nn.Conv2d
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

X = torch.rand(8, 8)
out = comp_conv2d(conv2d, X)
print(out, '\n', out.shape)
# 步幅均为2
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2)
out = comp_conv2d(conv2d, X)
print(out, '\n', out.shape)

"""
当输入的高和宽两侧的填充数分别为ph和pw时，我们称填充为(ph,pw)
当在高和宽上的步幅分别为sh和sw时，我们称步幅为(sh,sw)
"""
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
out = comp_conv2d(conv2d, X)
print(out, '\n', out.shape)
