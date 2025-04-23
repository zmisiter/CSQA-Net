import os

from settings.defaults import _C
from settings.setup_functions import *
import argparse  # 没必要使用argparse，做实验时可以调一个参数看一个结果

# 把丑陋的东西放到这里，保证main函数的整洁

config = _C.clone()
cfg_file = os.path.join('configs', 'cub', 'cub_vit.yaml')
# 连接多个路径名分量，若各组件名首字母没有/，则自动加/；https://blog.csdn.net/Hunter_Murphy/article/details/108037172
config = SetupConfig(config, cfg_file)
config.defrost()

# Log Name and Perferences
# config.write = True
# config.train.checkpoint = True  # f字符串自动调用参数里的名字
config.misc.exp_name = f'{config.data.dataset + config.model.type}'  # 到远程主机了，output中是按照数据集的名字来命名，数据集的名字在data_loader里面命名，名字有很多种，要提前定义好
config.misc.log_name = f'vit-b + msqe + lr = 16e-2'  # 每个实验要自己命名，比如+... 自动会给不带年份的日期以及时间，测试的时候可以根据时间把多余的删掉
# config.cuda_visible = '3,1,6,2'  # 179、197的服务器，这里每次都是必须要改的，每次跑之前都要看一下别人在用没用
config.cuda_visible = '1'   # 可见的gpu

# Environment Settings
config.data.log_path = os.path.join(config.misc.output, config.misc.exp_name, config.misc.log_name
                                    + time.strftime(' %m-%d_%H-%M', time.localtime()))  # 结果最终保存在哪里；以什么样的名字保存

config.model.pretrained = os.path.join(config.model.pretrained,    # pretrained/ResNet-50.pth(预训练模型的路径)
                                       config.model.name + config.model.pre_version + config.model.pre_suffix)
os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible
os.environ['OMP_NUM_THREADS'] = '1'

# Setup Functions
config.nprocess, config.local_rank = SetupDevice()
config.data.data_root, config.data.batch_size = LocateDatasets(config)  # 定位数据集，拿到数据集存放路径和batchsize
config.train.lr = ScaleLr(config)  # 定义学习率，打印出来的学习率是改变之后的(也就是写到论文中的，打印到终端表格里面的)
log = SetupLogs(config, config.local_rank)  # logs；外面写要用config.local_rank，函数内部写可以用rank
if config.write and config.local_rank in [-1, 0]:  # 写日志
    with open(config.data.log_path + '/config.json', "w") as f:  # 对应output中的config.json文件
        f.write(config.dump())  # dump方法用于将对象转换为字符串或字节流，并保存到文件或进行网络传输等操作
config.freeze()
SetSeed(config)


# if __name__ == '__main__':
#     print(config.local_rank)
