# This file contains modules common to various models

import torch
import torch.nn as nn
from models.common import Conv


class surrogate_silu(nn.Module):
    """docstring for surrogate_silu"""
    def __init__(self):
        super(surrogate_silu, self).__init__()
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x*self.act(x)


class surrogate_hardswish(nn.Module):
    """docstring for surrogate_hardswish"""
    def __init__(self):
        super(surrogate_hardswish, self).__init__()
        self.relu6 = nn.ReLU()

    def forward(self, x):
        return x *(self.relu6(torch.add(x, 3))/6)


class surrogate_focus(nn.Module):
    # surrogate_focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(surrogate_focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

        with torch.no_grad():
            self.conv1 = nn.Conv2d(3, 3, (2, 2), groups=3, bias=False, stride=(2, 2))
            self.conv1.weight[:, :, 0, 0] = 1
            self.conv1.weight[:, :, 0, 1] = 0
            self.conv1.weight[:, :, 1, 0] = 0
            self.conv1.weight[:, :, 1, 1] = 0

            self.conv2 = nn.Conv2d(3, 3, (2, 2), groups=3, bias=False, stride=(2, 2))
            self.conv2.weight[:, :, 0, 0] = 0
            self.conv2.weight[:, :, 0, 1] = 0
            self.conv2.weight[:, :, 1, 0] = 1
            self.conv2.weight[:, :, 1, 1] = 0

            self.conv3 = nn.Conv2d(3, 3, (2, 2), groups=3, bias=False, stride=(2, 2))
            self.conv3.weight[:, :, 0, 0] = 0
            self.conv3.weight[:, :, 0, 1] = 1
            self.conv3.weight[:, :, 1, 0] = 0
            self.conv3.weight[:, :, 1, 1] = 0

            self.conv4 = nn.Conv2d(3, 3, (2, 2), groups=3, bias=False, stride=(2, 2))
            self.conv4.weight[:, :, 0, 0] = 0
            self.conv4.weight[:, :, 0, 1] = 0
            self.conv4.weight[:, :, 1, 0] = 0
            self.conv4.weight[:, :, 1, 1] = 1

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)], 1))
