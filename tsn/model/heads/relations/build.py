# -*- coding: utf-8 -*-

"""
@date: 2020/9/6 下午7:56
@file: build.py
@author: zj
@description: 
"""

from .relation_module import RelationModule
from .multiscale_relation_module import RelationModuleMultiScale


def build_trn_module(cfg):
    num_frames = cfg.DATASETS.NUM_SEGS
    num_class = cfg.MODEL.HEAD.NUM_CLASSES
    relation_type = cfg.MODEL.HEAD.RELATION.NAME
    img_feature_dim = cfg.MODEL.HEAD.RELATION.IMG_FEATURE_DIMS

    if relation_type == 'TRN':
        TRNmodel = RelationModule(img_feature_dim, num_frames, num_class)
    elif relation_type == 'TRNmultiscale':
        TRNmodel = RelationModuleMultiScale(img_feature_dim, num_frames, num_class)
    else:
        raise ValueError('Unknown TRN' + relation_type)

    return TRNmodel
