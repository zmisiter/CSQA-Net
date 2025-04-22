import gc

import torch.nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from timm.utils import AverageMeter, accuracy
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from thop import profile
from models.build import build_models, freeze_backbone
from setup import config, log
from utils.data_loader import build_loader
from utils.eval import *
from utils.info import *
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
def build_model(config, num_classes):
	model = build_models(config, num_classes)  # model分为baseline的和自己定义的
	if torch.__version__[0] == 2:
		model = torch.compile(model, mode="max-autotune")
	model.to(device) # 把模型移动到gpu上面   model.cuda()
	freeze_backbone(model, config.train.freeze_backbone)  # build中把主干网络参数冻结
	model_without_ddp = model  # 如果使用ddp，模型会被嵌套一层，不能直接使用model.xxx调用参数；创建一个不受DDP影响的模型副本,为了保存模型方便
	n_parameters = count_parameters(model)
	config.defrost()  # 为了防止config被更改，先freeze锁上，需要更改时再解锁defrost
	config.model.num_classes = num_classes  # 用传入的类别数替换默认值
	config.model.parameters = f'{n_parameters:.3f}M'  # defaults中的parameters被这里修改
	config.freeze()
	if config.local_rank in [-1, 0]:  # 输出模型结构，这里对应最下面的model structure
		PSetting(log, 'Model Structure', config.model.keys(), config.model.values(),
				 rank=config.local_rank)  # 一个字符串对应一个值
		PSetting(log, 'ELPModel Structure', config.elp.keys(), config.elp.values(),
				 rank=config.local_rank)
		log.save(model)  # log是Log的实例化，所以它包含了其中定义的函数；这里是保存模型结构
	return model, model_without_ddp


def main(config):
	# Timer，计时器
	prepare_timer = Timer()  # 模型酝酿
	prepare_timer.start()  # 开始计时，一个timer、一个start，一个stop，stop的时候会返回一个值，这个值就是秒数
	train_timer = Timer()
	eval_timer = Timer()

	# Initialize the Tensorboard Writer
	writer = None
	if config.write:
		writer = SummaryWriter(config.data.log_path)

	# Prepare dataset
	train_loader, test_loader, num_classes, train_samples, test_samples, mixup_fn = build_loader(config)
	step_per_epoch = len(train_loader)  # 每个epoch要跑多少步；也就是多少个批次
	# print(len(train_loader))  1个gpu 187；2个gpu 93
	# for inputs, label in train_loader:
	#     print(inputs, label)
	total_batch_size = config.data.batch_size * get_world_size()
	steps = config.train.epochs * step_per_epoch  # 轮数×每轮要跑多少个iter

	# Build model 尽量保证每个可学习的参数都能用到，都能反传到梯度

	model, model_without_ddp = build_model(config, num_classes)
	model.cuda()

	if config.local_rank != -1:  # 如果用服务器跑，rank的值就不是-1，自动使用DDP
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],
														  broadcast_buffers=False,
														  find_unused_parameters=False)

	log.info("Calculate MACs & FLOPs ...")
	inputs = torch.randn((1, 3, config.data.img_size, config.data.img_size)).cuda()
	macs, num_params = profile(model, (inputs,), verbose=False)  # type: ignore
	log.info(
		"\nParams(M):{:.2f}, MACs(G):{:.2f}, FLOPs(G):~{:.2f}".format(num_params / (1000 ** 2), macs / (1000 ** 3),
		                                                              2 * macs / (1000 ** 3)))
	# 尽量保证每个可学习的参数都能用到，都能反传到梯度
	optimizer = build_optimizer(config, model, backbone_low_lr=True)
	loss_scaler = NativeScalerWithGradNormCount()  # AMP半精度，降低数据集的位数，提高精度，减少显存占用；自己搜了解一下
	# Build learning rate scheduler

	if config.train.lr_epoch_update:  # 每个epoch是否更新学习率，默认是False
		scheduler = build_scheduler(config, optimizer, 1)
	else:
		scheduler = build_scheduler(config, optimizer, step_per_epoch)

	# Determine criterion
	best_acc, best_epoch, train_accuracy = 0., 0., 0.

	if config.data.mixup > 0.:  # 车数据集要用mixup,也是数据增强的一种方法
		criterion = SoftTargetCrossEntropy()
	elif config.model.label_smooth:
		criterion = LabelSmoothingCrossEntropy(smoothing=config.model.label_smooth)
	else:
		criterion = torch.nn.CrossEntropyLoss()

	# Function Mode
	if config.model.resume:  # 负责加载的，如果是暂停模式，就会输出当前的最高精度
		best_acc = load_checkpoint(config, model_without_ddp, optimizer, scheduler, loss_scaler, log)
		best_epoch = config.train.start_epoch
		accuracy, loss = valid(config, model, test_loader, best_epoch, train_accuracy, writer, True)
		log.info(f'Epoch {best_epoch:^3}/{config.train.epochs:^3}: Accuracy {accuracy:2.3f}  '
				 f'BA {best_acc:2.3f}    BE {best_epoch:3}  '
				 f'Loss {loss:1.4f}    TA {train_accuracy * 100:2.2f}')
		if config.misc.eval_mode:  # 如果是评估模式，main函数就没了，因为如上面模型在恢复时会直接测试一遍
			return

	if config.misc.throughput:  # 只计算模型的吞吐量，把网络控制在可控的时间内；太慢的话就要去找原因：比如对400×400×32的特征图做卷积
		throughput(test_loader, model, log, config.local_rank)
		return

	# Record result in Markdown Table
	mark_table = PMarkdownTable(log, ['Epoch', 'Accuracy', 'Best Accuracy',
									  'Best Epoch', 'Loss'], rank=config.local_rank)

	# End preparation
	torch.cuda.synchronize()  # 在CUDA设备上同步操作，它会阻塞程序执行，直到之前所有在CUDA设备上的操作都完成
	prepare_time = prepare_timer.stop()  # 酝酿时间结束
	PSetting(log, 'Training Information',  # 打印
			 ['Train samples', 'Test samples', 'Total Batch Size', 'Load Time', 'Train Steps',
			  'Warm Epochs'],
			 [train_samples, test_samples, total_batch_size,
			  f'{prepare_time:.0f}s', steps, config.train.warmup_epochs],
			 newline=2, rank=config.local_rank)

	# Train Function
	sub_title(log, 'Start Training', rank=config.local_rank)  # 打印一堆等号.start training.一堆等号
	for epoch in range(config.train.start_epoch, config.train.epochs):  # 遍历epoch
		train_timer.start()  # 封装训练阶段
		model.train(True)

		if config.local_rank != -1:  # 如果用DDP训练,数据集在每个epoch的采样是随机的,shuffle操作
			train_loader.sampler.set_epoch(epoch)
		if not config.misc.eval_mode:
			# list1 = list(model.named_parameters())
			# print(list1[53])
			train_accuracy = train_one_epoch(config, model, criterion, train_loader, optimizer,
											  epoch, scheduler, loss_scaler, mixup_fn, writer)  # 得到一个训练精度
			# if config.local_rank in [-1, 0]:
			#     try:
			#         model.get_count()
			#     except:
			#         model.module.get_count()
			# print(model.module.a1)
		train_timer.stop()

		# Eval Function
		eval_timer.start()
		# 2）当epoch+1能被eval_every整除 3）到最后一个epoch；这些时候对模型进行测试
		if epoch < 5 or (epoch + 1) % config.misc.eval_every == 0 or epoch + 1 == config.train.epochs:
			accuracy, loss = valid(config, model, test_loader, epoch, train_accuracy, writer)  # 做验证
			if config.local_rank in [-1, 0]:  # 只在本地或者第一块gpu的进程打印
				if best_acc < accuracy:
					best_acc = accuracy
					best_epoch = epoch + 1
					if config.write and epoch > 10 and config.train.checkpoint:  # 如果达到最高精度，就把最高精度存起来，写一下
						save_checkpoint(config, epoch, model, best_acc, optimizer, scheduler, loss_scaler, log)
				log.info(
					f'Epoch {epoch + 1:^3}/{config.train.epochs:^3}: Accuracy {accuracy:2.3f}  '  # 不管达不达到最高精度，都输出一下
					f'BA {best_acc:2.3f}    BE {best_epoch:3}   '
					f'Loss {loss:1.4f}    TA {train_accuracy * 100:2.2f}')
				if config.write:  # 测试时可能报错很多次，不希望output一大堆，那就可以把写先关掉，能跑通几个epoch再把写打开，写日志
					mark_table.add(log, [epoch + 1, f'{accuracy:2.3f}',
										 f'{best_acc:2.3f}', best_epoch, f'{loss:1.5f}'], rank=config.local_rank)
			pass  # Eval
		eval_timer.stop()
		pass  # Train

	torch.cuda.empty_cache()

	# Finish Training
	if writer is not None:
		writer.close()
	train_time = train_timer.sum / 60
	eval_time = eval_timer.sum / 60
	total_time = train_time + eval_time
	PSetting(log, "Finish Training",  # 这里对应终端最后的输出
			 ['Best Accuracy', 'Best Epoch', 'Training Time', 'Testing Time', 'Total Time'],
			 [f'{best_acc:2.3f}', best_epoch, f'{train_time:.2f} min', f'{eval_time:.2f} min', f'{total_time:.2f} min'],
			 newline=2, rank=config.local_rank)  # 每两个一换行
	pass


def train_one_epoch(config, model, criterion, train_loader, optimizer,
					epoch, scheduler, loss_scaler, mixup_fn=None, writer=None):
	# optimizer.zero_grad()

	step_per_epoch = len(train_loader)  # 批次数量
	loss_meter = AverageMeter()
	norm_meter = AverageMeter()
	scaler_meter = AverageMeter()

	loss_img_meter = AverageMeter()
	loss_parts_meter = AverageMeter()
	# loss_reg_meter = AverageMeter()

	epochs = config.train.epochs
	p_bar = tqdm(total=step_per_epoch,
				 desc=f'Train {epoch + 1:^3}/{epochs:^3}',  # epoch是在for循环中设定的值
				 dynamic_ncols=True,  # 动态调整进度条的宽度，使其适应终端窗口的大小
				 ascii=True,  # 使用ASCII字符显示进度条，这意味着进度条将使用文本字符进行绘制
				 disable=config.local_rank not in [-1, 0])  # 根据config.local_rank的值决定是否禁用进度条
	all_preds, all_label = None, None
	for step, (x, y) in enumerate(train_loader):
		model.train(True)
		global_step = epoch * step_per_epoch + step  # 步数即批次数
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)  # 数据和标签转到gpu上面，复制是异步的
		# y = torch.from_numpy(np.array(y)).cuda(non_blocking=True)

		optimizer.zero_grad()
		if mixup_fn:  # 是否使用条件混合
			x, y = mixup_fn(x, y)
		with torch.cuda.amp.autocast(enabled=config.misc.amp):  # 用amp，这里有点反直觉
			if config.model.baseline_model:  # 模型计算好了损失，如果想要加损失，就提前算好输出去？
				logits = model(x)  # 标准的resnet50不支持输入标签，所以这里只有x
			else:
				logits = model(x, y, epoch, step, step_per_epoch, config.model.no_elp, config.model.no_class)  # 这里传epoch,因为对应原图可以看出有四个损失,在main函数里计算就需要写很多东西很麻烦,那就希望损失在模型里已经计算好了

		logits, loss, other_loss = loss_in_iters(logits, y, criterion)  # 判断从model的输出是一个还是两个，对应model中的损失

		# loss_img = other_loss[0]
		# loss_parts = other_loss[1]
		# loss_reg = other_loss[2]

		is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
		# gradnorm或者amp里面的
		if not config.model.baseline_model:
			grad_norm = loss_scaler(loss, optimizer, clip_grad=config.train.clip_grad,
									parameters=model.parameters(), create_graph=is_second_order,retain_graph=True)
		else:
			grad_norm = loss_scaler(loss, optimizer, clip_grad=config.train.clip_grad,
									parameters=model.parameters(), create_graph=is_second_order)

		# optimizer.zero_grad()  # 优化器梯度清零，以接收下一批梯度
		if config.train.lr_epoch_update:
			scheduler.step_update(epoch + 1)
		else:
			scheduler.step_update(global_step + 1)  # 每个训练步骤结束后,调用学习率调度器来更新学习率
		loss_scale_value = loss_scaler.state_dict()["scale"]  # 输出可视化,调出来,这里应该是grad_norm.state_dict()么？

		if mixup_fn is None:  # 用了mixup是没法获得训练精度的，只能看测试精度，因为mixup会把各个类别标签混合，没法判断对还是错
			preds = torch.argmax(logits, dim=-1)  # 存训练得到的标签和结果
			all_preds, all_label = save_preds(preds, y, all_preds, all_label)
		torch.cuda.synchronize()  # 等待当前设备上所有流中的所有核心完成；做线程同步，也就是等待kernel中所有线程全部执行完毕再执行CPU端后续指令

		# grad_norm表示计算得到的梯度范数，用于衡量梯度的大小；在后续的代码中，可能会使用grad_norm来进行梯度更新或其他相关的计算
		if grad_norm is not None:
			norm_meter.update(grad_norm)  # 用于记录梯度范数的指标
		scaler_meter.update(loss_scale_value)  # 用于记录缩放因子的指标
		loss_meter.update(loss.item(), y.size(0))  # 一个batch中的损失和其对应的样本数量

		# loss_img_meter.update(loss_img.item(), y.size(0))
		# loss_parts_meter.update(loss_parts.item(), y.size(0))
		# loss_reg_meter.update(loss_reg.item(), y.size(0))

		lr = optimizer.param_groups[2]['lr']  # 从第一个参数组中，通过键'lr'获取学习率的值
		optimizer.param_groups[0]['lr'] = config.elp.lr
		# print(optimizer.param_groups[0]['lr'],optimizer.param_groups[1]['lr'],optimizer.param_groups[2]['lr'])
		if writer:
			writer.add_scalar("train/loss", loss_meter.val, global_step)
			writer.add_scalar("train/lr", lr, global_step)
			writer.add_scalar("train/grad_norm", norm_meter.val, global_step)
			writer.add_scalar("train/scaler_meter", scaler_meter.val, global_step)
			if other_loss:
				try:
					loss_img_meter.update(other_loss[0].item(), y.size(0))
					loss_parts_meter.update(other_loss[1].item(), y.size(0))
					# loss_reg_meter.update(other_loss[2].item(), y.size(0))
				except:
					pass

				writer.add_scalar("losses/t_loss", loss_meter.val, global_step)
				writer.add_scalar("losses/1_loss", loss_img_meter.val, global_step)
				writer.add_scalar("losses/2_loss", loss_parts_meter.val, global_step)
				# writer.add_scalar("losses/3_loss", loss_reg_meter.val, global_step)

		p_bar.set_postfix(loss="%2.5f" % loss_meter.avg, lr="%.5f" % lr, gn="%1.4f" % norm_meter.avg)
		p_bar.update()
		torch.cuda.empty_cache()

	# After Training an Epoch
	p_bar.close()
	# if not config.model.baseline_model and config.local_rank in [-1, 0]:
	#     log.info(
	#         f'Loss {loss_img_meter.avg:2.3f}  {loss_parts_meter.avg:2.3f}  {loss_reg_meter.avg:2.3f}'
	#         )
	train_accuracy = eval_accuracy(all_preds, all_label, config) if mixup_fn is None else 0.0
	gc.collect()
	return train_accuracy


def loss_in_iters(output, targets, criterion):
	if not isinstance(output, (list, tuple)):
		return output, criterion(output, targets), None
	else:
		logits, loss = output
		if not isinstance(loss, (list, tuple)):
			return logits, loss, None
		else:
			return logits, loss[0], loss[1:]

@torch.no_grad()  # 不进行梯度计算，以节省内存和计算资源
def valid(config, model, test_loader, epoch=-1, train_acc=0.0, writer=None, save_feature=True):
	criterion = torch.nn.CrossEntropyLoss()  # 标准，交叉熵损失
	model.eval()

	step_per_epoch = len(test_loader)
	p_bar = tqdm(total=step_per_epoch,
				 desc=f'Valid {(epoch + 1) // config.misc.eval_every:^3}/{math.ceil(config.train.epochs / config.misc.eval_every):^3}',
				 # 进度条的前缀，前者是结果取整，居中对齐占3个字符；后者是结果向上取整
				 dynamic_ncols=True,
				 ascii=True,
				 disable=config.local_rank not in [-1, 0])

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	saved_feature,saved_labels=[],[]
	for step, (x, y) in enumerate(test_loader):
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

		with torch.cuda.amp.autocast(enabled=config.misc.amp):  # 用上下文管理器来自动混合精度计算
			logits = model(x)  # 不管训练还是测试，输出是一个还是两个，一定输出的是没有标签的；如果baseline模块中，x就够了，y是在外面算的；测试的时候只有x

		logits, loss, _ = loss_in_iters(logits, y, criterion)

		acc = accuracy(logits, y)[0]
		if config.local_rank != -1:
			acc = reduce_mean(acc)

		if save_feature:
			saved_feature.append(logits)
			saved_labels.append(y)

		loss_meter.update(loss.item(), y.size(0))
		acc_meter.update(acc.item(), y.size(0))

		p_bar.set_postfix(acc="{:2.3f}".format(acc_meter.avg), loss="%2.5f" % loss_meter.avg,
						  tra="{:2.3f}".format(train_acc * 100))  # 评估模式进度条显示的东西
		p_bar.update()
		pass

	if save_feature:
		os.makedirs('visualize/saved_features', exist_ok=True)
		saved_feature = torch.cat(saved_feature, 0)
		saved_labels = torch.cat(saved_labels, 0)
		torch.save(saved_feature, f'visualize/saved_features/{config.data.dataset}_f.pth')
		torch.save(saved_labels, f'visualize/saved_features/{config.data.dataset}_l.pth')

	p_bar.close()
	if writer:
		writer.add_scalar("test/accuracy", acc_meter.avg, epoch + 1)
		writer.add_scalar("test/loss", loss_meter.avg, epoch + 1)
		writer.add_scalar("test/train_acc", train_acc * 100, epoch + 1)
	return acc_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, log, rank):  # 吞吐量
	model.eval()
	for idx, (images, _) in enumerate(data_loader):
		images = images.cuda(non_blocking=True)
		batch_size = images.shape[0]
		for i in range(50):
			model(images)
		torch.cuda.synchronize()
		if rank in [-1, 0]:
			log.info(f"throughput averaged with 30 times")
		tic1 = time.time()
		for i in range(30):
			model(images)
		torch.cuda.synchronize()
		tic2 = time.time()
		if rank in [-1, 0]:
			log.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
		return


if __name__ == '__main__':
	if config.local_rank ==-1:
		config.defrost()
		config.write = False
		config.freeze()
	main(config)

