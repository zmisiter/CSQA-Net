# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch
from torch import optim as optim


def build_optimizer(config, model, backbone_low_lr=True):
    # SGD或者ADAMW；backbone_low_lr代表的是更新主干网络参数的时候，学习率会比较低；正常的学习率去更新自己新加的网络
    """Build optimizer, set weight decay of normalization to 0 by default."""
    skip = {}  # 创建空字典
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):  # hasattr() 函数用于判断对象是否包含对应的属性。
        skip = model.no_weight_decay()  # 有的用权重衰减、有的没用权重衰减
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    if backbone_low_lr:
        parameters = set_elp_lr(model)
    else:
        parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.train.optimizer.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.train.momentum, nesterov=True,  # 动量加快梯度下降的速度
                              lr=config.train.lr, weight_decay=config.train.weight_decay)

    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.train.eps, betas=config.train.betas,
                                lr=config.train.lr, weight_decay=config.train.weight_decay)
    return optimizer


def set_elp_lr(model):
    with_elp = []
    has_backbone= []
    without_elp = []
    with_attention = []
    for name, param in model.named_parameters():
        if 'elp_list' in name:
            with_elp.append(param)
            # print(f'ELP{name}')
        elif 'attention6' in name:
            with_attention.append(param)
        elif 'backbone' in name:
            has_backbone.append(param)
            # print(f'Backbone{name}')
        else:
            without_elp.append(param)
            # print(f'Other{name}')
    return [{'params': with_elp},  # 新加的模块参数使用默认的学习率更新
            {'params': with_attention, 'lr_scale': 0.05}, # , 'lr_scale': 0.05
            {'params': without_elp},
            {'params': has_backbone, 'lr_scale': 0.1}]   # , 'lr_scale': 0.1
            # 主干网络参数使用默认学习率的0.1倍


def set_weight_decay(model, skip_list=(), skip_keywords=()):  # 同时实现主干网络低学习率和权重衰减的分割
    has_decay = []  # 初始化列表
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
