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


"""
结果中保留行和列这两个维度（keepdim=True）

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))  # dim=0 同一列求和
print(X.sum(dim=1, keepdim=True))  # dim=1 同一行求和
"""

#  获取和读取数据
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(True)
b.requires_grad = True
print(W.requires_grad)


#  一分钟理解softmax函数 https://blog.csdn.net/lz_peter/article/details/84574716
def softmax(X):
    X_exp = X.exp()  # Exp 函数 返回 e（自然对数的底）的幂次方。保证了概率的非负性
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)  # 矩阵相乘


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0, 0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()


X = torch.rand((2, 5))
X_prob = softmax(X)
# print(X_prob, X_prob.sum(dim=1))

y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))
# print(y_hat.gather(1, y.view(-1, 1)))

print(accuracy(y_hat, y))
print(evaluate_accuracy(test_iter, net))
