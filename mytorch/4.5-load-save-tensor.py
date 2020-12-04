#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
created by Halo 2020/11/13 11:29
"""

"""
在实际中，我们有时需要把训练好的模型部署到很多不同的设备。
在这种情况下，我们可以把内存中训练好的模型参数存储在硬盘上供后续读取使用。
"""

import torch
from torch import nn

x = torch.ones(3)
torch.save(x, 'x.pt')

x2 = torch.load('x.pt')
print(x2)

y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)

torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


net = MLP()
net.state_dict()
print(net.state_dict())

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer.state_dict()
print(optimizer.state_dict())

torch.save(net.state_dict(), 'models.pt')
net = MLP()
net.load_state_dict(torch.load('models.pt'))
print(net)
