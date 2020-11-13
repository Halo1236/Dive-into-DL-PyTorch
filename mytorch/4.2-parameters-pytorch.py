#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
created by Halo 2020/11/10 11:23
"""
# 4.2 模型参数的访问、初始化和共享
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import init

# pytorch已进行默认初始化
net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()

print(type(net.named_parameters()))

for name, param in tqdm(net.named_parameters()):
    print(name, '-', param.data)

for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))


class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)

    def forward(self, x):
        pass


n = MyModel()
for name, param in n.named_parameters():
    print(name, param)

for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)


def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()


for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

'''
Module类的forward函数里多次调用同一个层。
此外，如果我们传入Sequential的模块是同一个Module实例的话参数也是共享的，下面来看一个例子:
'''
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))

x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)  # 单次梯度是3，两次所以就是6
