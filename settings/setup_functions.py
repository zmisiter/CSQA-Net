import random
import socket
import torch.nn as nn

import numpy as np
import yaml

from utils.eval import get_world_size
from utils.info import *
from settings.defaults import _C

config = _C.clone()


# setup_functions里面是一些加载函数
def SetupConfig(config, cfg_file=None):  # 加载yaml，把yaml和defaults作合并
    if cfg_file:
        config.defrost()  # 打开修改权限
        print('-' * 28, '{:^22}'.format(cfg_file), '-' * 28)  # https://zhuanlan.zhihu.com/p/60357679
        config.merge_from_file(cfg_file)  # https://zhuanlan.zhihu.com/p/366289700#
        config.freeze()
    return config


def get_topk_tokens(tokens, topk_indices):
    # tokens: (B, N-1, D)
    # topk_indices: (B, topk)
    B, N_minus_1, D = tokens.shape

    # Expand indices to have the correct shape for advanced indexing
    topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, topk, D)

    # Use gather to select the topk tokens
    topk_tokens = torch.gather(tokens, dim=1, index=topk_indices)

    return topk_tokens

def SetupLogs(config, rank=0):  # 传统的方法只会在终端里输出，没法写出去；这里可以同时写，也可以输出到终端里去；之前是需要copy一下会很麻烦
    write = config.write  # 根据setup中的值改变，是否写入
    if rank not in [-1, 0]:
        return
    # 所有打印的地方都需要加，只打印一份并且在0号进程打印，或者是自己电脑打印输出，比如4张卡是4个独立的进程，如果不加if rank，每个进程都会调用打印的函数，会打印4份
    # 建立log对象
    if write:  # 若写入，写入的位置是在setup里面定义的
        os.makedirs(config.data.log_path, exist_ok=True)  # 用来创建多层目录，path是需要递归创建的目录；exist=true在目录已存在的情况下不会触发异常
    log = Log(fname=config.data.log_path, write=write)  # 调用日志的，包括终端和文件；两种都可以写,实例化Log类
    # 分类打印输出，就是终端输出的样子
    # PTitle(log,config.local_rank)    这里可以添加自己想要的标题；终端输出的形式为键值对，函数外面要写config.local_rank，里面可以直接写rank
    PSetting(log, 'Data Settings', config.data.keys(), config.data.values(), newline=2,
             rank=config.local_rank)  # newline是每两个参数一换行
    PSetting(log, 'Hyper Parameters', config.parameters.keys(), config.parameters.values(), rank=config.local_rank)
    PSetting(log, 'Training Settings', config.train.keys(), config.train.values(), rank=config.local_rank)
    PSetting(log, 'Other Settings', config.misc.keys(), config.misc.values(), rank=config.local_rank)

    # 返回log实例
    return log


def SetupDevice():  # 'RANK'和'WORLD_SIZE'是在系统环境里面用字符串表示的值，用int函数转换成值，设置DDP的一些必要步骤
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:  # 判断两个环境变量是否在系统环境中，WORLD_SIZE是未屏蔽掉的gpu总数，rank是重新给的编号
        rank = int(os.environ["RANK"])  # 通过os.environ[""]获取有关系统的各种信息，根据一个字符串映射到系统环境的一个对象，用int函数取到值并赋给rank
        world_size = int(os.environ['WORLD_SIZE'])  # world_size是一个机器卡的数量还是所有机器卡的数量暂时未知；比如4块卡，world_size=4
        torch.cuda.set_device(rank)  # 这里重新排列的编号就是0,1,2,3
        # torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size,
        #                                      rank=rank)  # windows下是这样的
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size,
                                             rank=rank)  # linux下是这样的
        torch.distributed.barrier()  # 保证进程同步，所有的显卡都运算完了才会进行下一步
    else:  # 在本地
        rank = -1
        world_size = -1
    nprocess = torch.cuda.device_count()  # 返回gpu可用数量
    torch.cuda.set_device(rank)
    # torch.backends.cudnn.benchmark = True
    return nprocess, rank


def SetSeed(config):  # 电脑里的随机是伪随机，有一张随机的码表(内含许多的随机数)，每次是随机调一个出来;
    seed = config.misc.seed + config.local_rank
    # 种子的作用就是在同样的参数下，同样的编译器版本下，代码具有可重复性；后面+的local_rank就是为了让每张显卡上的种子都是固定但都不一样，以此来增强特征多样性
    torch.manual_seed(seed)  # 为cpu设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前gpu设置随机种子
    np.random.seed(seed)  # 如果做了数据预处理，那么对python、numpy的随机数生成器也需要设置种子
    random.seed(seed)


def ScaleLr(config):
    base_lr = config.train.lr * config.data.batch_size * get_world_size() / 512.0   # 512.0
    # 省去了batchsize变小，而要手动调小lr的步骤，这里就是打印出来的学习率
    return base_lr


def LocateDatasets(config):  # 之前每次跑都要选服务器的名字，不然都不知道数据集的文件夹在哪；自创的，让电脑自动判断该用哪个数据集、数据集的路径是什么
    # config = None
    def HostIp():
        """ 查询本机ip地址: return: ip"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP Socket
            s.connect(('8.8.8.8', 80))  # 连接到address处的套接字，一般address的格式为元组（hostname,port）
            ip = s.getsockname()[0]  # 返回套接字自己的地址，通常是一个元组(ipaddr,port)
        finally:
            s.close()  # finally都会执行，关闭套接字
        return ip

    ip = HostIp()
    print(ip)
    address = ip.split('.')[3]  # 用.分割ip地址，并取第四个位置也就是末端的字符赋给address
    data_root = config.data.data_root  # 自动改变数据集的路径和batch_size
    batch_size = config.data.batch_size
    if address == '179':
        data_root = '/DATA/meiyiming/ly/dataset'
        batch_size = config.data.batch_size // 2
    elif address == '197':
        data_root = '/DATA/linjing/ly/dataset'
        batch_size = config.data.batch_size // 2  # 197和179的显存都为12G，所以向下整除4
    elif address == '227':
        data_root = 'F:\\Datasets'  # 本地显存
        batch_size = config.data.batch_size // 2
    elif address == '118':
        data_root = '/home/cvpr/dataset/'  # 本地显存
        batch_size = config.data.batch_size
    # 组里的服务器ip地址来回变，并且显存为24G，所以不整除
    return data_root, batch_size


if __name__ == '__main__':
    LocateDatasets(config)
    nprocess, rank = SetupDevice()
    print(rank)
    print(type(rank))
