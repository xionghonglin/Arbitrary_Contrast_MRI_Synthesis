import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class ResNet(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(ResNet, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        self.out_dim = n_feats


    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x
        return res


@register('ResNet-16-64')
def make_encoder(n_resblocks=16, n_feats=64, res_scale=1, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.rgb_range = rgb_range
    args.n_colors = 1
    return ResNet(args)

@register('ResNet-24-128')
def make_encoder(n_resblocks=24, n_feats=128, res_scale=1, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.rgb_range = rgb_range
    args.n_colors = 1
    return ResNet(args)

@register('ResNet-32-256')
def make_encoder(n_resblocks=32, n_feats=256, res_scale=0.1, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.rgb_range = rgb_range
    args.n_colors = 1
    return ResNet(args)