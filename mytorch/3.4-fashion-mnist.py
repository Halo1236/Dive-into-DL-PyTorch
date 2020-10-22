#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
created by Halo 2020/10/21 16:48
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import sys

sys.path.append("..")
import mytorch.d2lzh_pytorch as d2l

"""
另外我们还指定了参数transform = transforms.ToTensor()使所有数据转换为Tensor，如果不进行转换则返回的是PIL图片。transforms.ToTensor()将尺寸为 (H x W x C) 
且数据位于[0, 255]的PIL图片或者数据类型为np.uint8的NumPy数组转换为尺寸为(C x H x W)且数据类型为torch.float32且位于[0.0, 1.0]的Tensor。 
"""


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def load_data_fashion_mnist(batch_size, root='../Datasets/FashionMNIST'):
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True,
                                                   transform=transforms.ToTensor())
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


#  获取和读取数据
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)))
b = torch.zeros(num_outputs, dtype=torch.float)

b.requires_grad = True
print(w.requires_grad)
