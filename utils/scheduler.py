# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
from timm.scheduler import MultiStepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler
from timm.scheduler.step_lr import StepLRScheduler


def build_scheduler(config, optimizer, step_per_epoch):
    num_steps = int(config.train.epochs * step_per_epoch)  # 总步数
    warmup_steps = int(config.train.warmup_epochs * step_per_epoch)  # 预热步数
    lr_scheduler = None
    # lr_min = config.train.lr * 1e-2
    lr_min = config.train.lr * 0
    warmup_lr = config.train.lr * 1e-3  # 预热学习率  这里怎么和显示的对不上
    if config.train.scheduler == 'cosine':  # 学习率衰减，一般都是cos衰减
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,  # 学习率将在这些步数内进行调度
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            warmup_prefix=True,
            cycle_limit=1,  # 周期数为1，学习率将从初始学习率逐渐减小到最小学习率，然后保持最小学习率不变
            t_in_epochs=False,
        )
    elif config.train.scheduler == 'linear':  # 线性衰减
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.train.scheduler == 'step':  # step衰减
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=15,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


# def build_scheduler(config, optimizer, step_per_epoch):
# 	num_steps = int(config.train.epochs * step_per_epoch)
# 	warmup_steps = int(config.train.warmup_epochs * step_per_epoch)
#
# 	lr_scheduler = None
# 	# lr_min = config.train.lr * 1e-2
# 	# warmup_lr = config.train.lr * 1e-3
# 	if config.train.scheduler == 'cosine':
# 		lr_scheduler = WarmupCosineSchedule(optimizer,warmup_steps,num_steps)
# 	elif config.train.scheduler == 'linear':
# 		lr_scheduler = WarmupLinearSchedule(optimizer,warmup_steps,num_steps)
# 	elif config.train.scheduler == 'step':
# 		lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_per_epoch*15,0.1)
# 	return lr_scheduler

class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


import math

from torch.optim.lr_scheduler import LambdaLR


class ConstantLRSchedule(LambdaLR):
    # Constant learning rate schedule.

    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warm_steps, last_epoch=-1):
        self.warmup_steps = warm_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warm_steps, t_total, last_epoch=-1):
        self.warmup_steps = warm_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warm_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warm_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
                 warm_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warm_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                    self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr
