# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午10:57
@file: ucf101.py
@author: zj
@description: 
"""

import os

from .base_dataset import BaseDataset


class UCF101(BaseDataset):

    def __init__(self, data_dir, annotation_dir, modality=("RGB"), num_seg=3, split=1, train=True, transform=None):
        super(UCF101, self).__init__(data_dir, modality=modality, num_seg=num_seg, transform=transform)

        if train:
            annotation_path = os.path.join(annotation_dir, f'ucf101_train_split_{split}_rawframes.txt')
        else:
            annotation_path = os.path.join(annotation_dir, f'ucf101_val_split_{split}_rawframes.txt')

        if not os.path.isfile(annotation_path):
            raise ValueError(f'{annotation_path}不是文件路径')

        self.update(annotation_path)