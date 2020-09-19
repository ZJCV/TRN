# -*- coding: utf-8 -*-

"""
@date: 2020/9/19 下午3:18
@file: trn_head.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from tsn.model import registry
from .relations.build import build_trn_module


@registry.HEAD.register('TRNHead')
class TRNHead(nn.Module):

    def __init__(self, cfg):
        super(TRNHead, self).__init__()

        in_channels = cfg.MODEL.HEAD.FEATURE_DIMS
        img_feature_dims = cfg.MODEL.HEAD.RELATION.IMG_FEATURE_DIMS

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, img_feature_dims)
        self.relation = build_trn_module(cfg)

        self.num_segs = cfg.DATASETS.NUM_SEGS
        self.img_feature_dim = img_feature_dims
        self.num_class = cfg.MODEL.HEAD.NUM_CLASSES

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x).reshape(-1, self.num_segs, self.img_feature_dim)
        x = self.relation(x).reshape(-1, self.num_class)

        return x
