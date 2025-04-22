import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist


# eval里面是一些评估的函数

class Timer:  # 计时器
	def __init__(self):
		self.times = []
		self.start()
		self.avg = 0.
		self.count = 0.
		self.sum = 0.

	def start(self):
		self.tik = time.time()

	def stop(self):
		t = time.time() - self.tik
		self.times.append(t)
		self.sum += t
		self.count += 1
		self.avg = self.sum / self.count
		return self.times[-1]

	def cumsum(self):
		return np.array(self.times).cumsum().tolist()


def simple_accuracy(preds, labels):  # 简易的准确度
	count = preds.shape[0]
	result = (preds == labels).sum()  # (preds == labels).sum()
	return result / count


def reduce_mean(tensor):  # 多卡里面如何平均精度，如两张卡，1个batch分到两个显卡里，每个显卡里有两个样本；第一张卡正确率100，第二张卡0，则平均为50
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)  # 进行全局归约操作，后边的参数是将所有进程中的值相加
	rt /= get_world_size()
	return rt


def count_parameters(model):  # 计算参数
	params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # model.parameters()保存的是Weights和Bais参数的值
	return params / 1000000


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                    logger):  # 保存检查点，加载时，调用加载函数就可以了
	save_state = {'model': model.state_dict(),
	              'optimizer': optimizer.state_dict(),
	              'lr_scheduler': lr_scheduler.state_dict(),
	              'max_accuracy': max_accuracy,
	              'scaler': loss_scaler.state_dict(),
	              'epoch': epoch,
	              'config': config}

	# save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
	save_path = os.path.join(config.data.log_path, "checkpoint.bin")  # 这里的config.data.log_path没有在参数中定义好
	torch.save(save_state, save_path)
	print("----- Saved model checkpoint to", config.data.log_path, '-----')


def save_preds(preds, y, all_preds=None, all_label=None, ):  # 保存预测结果
	if all_preds is None:  # 如果为空，则表示第一次调用该函数
		all_preds = preds.clone().detach()
		all_label = y.clone().detach()
	else:  # 之前已经调用过该函数并保存有预测结果和真实标签
		all_preds = torch.cat((all_preds, preds), 0)
		all_label = torch.cat((all_label, y), 0)
	return all_preds, all_label


def load_checkpoint(config, model, optimizer=None, scheduler=None, loss_scaler=None, log=None):  # 加载检查点
	log.info(f"--------------- Resuming form {config.model.resume} ---------------")
	checkpoint = torch.load(config.model.resume, map_location='cpu')
	msg = model.load_state_dict(checkpoint['model'], strict=False)
	log.info(msg)
	max_accuracy = 0.0
	if not config.misc.eval_mode and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['lr_scheduler'])
		config.defrost()
		# config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
		config.freeze()
		if 'scaler' in checkpoint:
			loss_scaler.load_state_dict(checkpoint['scaler'])
		log.info(f"----- loaded successfully '{config.model.resume}' -- epoch {checkpoint['epoch']} -----")
		if 'max_accuracy' in checkpoint:
			max_accuracy = checkpoint['max_accuracy']

	del checkpoint
	torch.cuda.empty_cache()
	return max_accuracy


def eval_accuracy(all_preds, all_label, config):  # 两个函数的汇总，在简易精度的基础上加上了reduce_mean精度
	accuracy = simple_accuracy(all_preds, all_label)
	if config.local_rank != -1:
		# dist.barrier(device_ids=[config.local_rank])   # (这里单卡不用) 同步不同进程之间的操作，以确保在计算准确率之前，所有进程都达到了该位置
		val_accuracy = reduce_mean(accuracy)
	else:
		val_accuracy = accuracy
	return val_accuracy.item()  # 取回标量值item


class NativeScalerWithGradNormCount:
	state_dict_key = "amp_scaler"

	def __init__(self):
		self._scaler = torch.cuda.amp.GradScaler()

	def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,
				 retain_graph = True):
		self._scaler.scale(loss).backward(create_graph=create_graph,retain_graph=retain_graph)
		if update_grad:
			if clip_grad is not None:
				assert parameters is not None
				self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
				norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
			else:
				self._scaler.unscale_(optimizer)
				norm = ampscaler_get_grad_norm(parameters)
			self._scaler.step(optimizer)
			self._scaler.update()
		else:
			norm = None
		return norm

	def state_dict(self):
		return self._scaler.state_dict()

	def load_state_dict(self, state_dict):
		self._scaler.load_state_dict(state_dict)


def ampscaler_get_grad_norm(parameters,  # 小数点左移之后，计算出的损失可能很大，但是实际的损失很小，所以要把小数点再移回来
                            norm_type: float = 2.0) -> torch.Tensor:  # norm_type表示范数的类型
	if isinstance(parameters, torch.Tensor):
		parameters = [parameters]
	parameters = [p for p in parameters if p.grad is not None]  # 过滤掉没有梯度的参数
	norm_type = float(norm_type)
	if len(parameters) == 0:
		return torch.tensor(0.)
	device = parameters[0].grad.device  # 获取第一个参数的梯度所在的设备
	if norm_type == math.inf:
		total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)  # 获取参数列表中最大梯度值，并移动到gpu设备上
	else:
		total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
		                                                norm_type).to(device) for p in parameters]), norm_type)
		# 将每个参数的梯度范数放入一个张量列表中，并将其堆叠成一个张量，维度为(num_params,)
		# 再次使用torch.norm() 函数计算堆叠后的张量的范数，使用norm_type参数指定范数的类型
	return total_norm


def get_world_size():  # 获取参与训练的进程数量
	if not dist.is_available():  # 检查是否支持分布式训练
		return 1
	if not dist.is_initialized():
		return 1
	return dist.get_world_size()  # 说明dist模块可用且已初始化
