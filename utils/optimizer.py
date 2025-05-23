# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch
from torch import optim as optim


def build_optimizer(config, model, backbone_low_lr=True):
    """Build optimizer, set weight decay of normalization to 0 by default."""
    skip = {}  
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):  
        skip = model.no_weight_decay() 
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    if backbone_low_lr:
        parameters = set_qp_lr(model)
    else:
        parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.train.optimizer.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.train.momentum, nesterov=True, 
                              lr=config.train.lr, weight_decay=config.train.weight_decay)

    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.train.eps, betas=config.train.betas,
                                lr=config.train.lr, weight_decay=config.train.weight_decay)
    return optimizer


def set_qp_lr(model):
    with_qp = []
    has_backbone= []
    without_qp = []
    with_attention = []
    for name, param in model.named_parameters():
        if 'qp_list' in name:
            with_qp.append(param)
            # print(f'qp{name}')
        elif 'attention6' in name:
            with_attention.append(param)
        elif 'backbone' in name:
            has_backbone.append(param)
            # print(f'Backbone{name}')
        else:
            without_qp.append(param)
            # print(f'Other{name}')
    return [{'params': with_qp},  
            {'params': with_attention, 'lr_scale': 0.05}, # , 'lr_scale': 0.05
            {'params': without_qp},
            {'params': has_backbone, 'lr_scale': 0.1}]   # , 'lr_scale': 0.1
          


def set_weight_decay(model, skip_list=(), skip_keywords=()):  
    has_decay = []  
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
