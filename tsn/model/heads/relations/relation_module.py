# -*- coding: utf-8 -*-

"""
@date: 2020/9/6 下午7:51
@file: relation_module.py
@author: zj
@description: 
"""

import torch.nn as nn


class RelationModule(nn.Module):

    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.classifier = self.fc_fusion()

    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 512
        classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
            nn.ReLU(),
            nn.Linear(num_bottleneck, self.num_class),
        )
        return classifier

    def forward(self, input):
        input = input.view(input.size(0), self.num_frames * self.img_feature_dim)
        input = self.classifier(input)
        return input
