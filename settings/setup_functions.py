import random
import socket
import torch.nn as nn

import numpy as np
import yaml

from utils.eval import get_world_size
from utils.info import *
from settings.defaults import _C

config = _C.clone()


def SetupConfig(config, cfg_file=None):  
    if cfg_file:
        config.defrost()  
        print('-' * 28, '{:^22}'.format(cfg_file), '-' * 28)  
        config.merge_from_file(cfg_file) 
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

def SetupLogs(config, rank=0):  
    write = config.write 
    if rank not in [-1, 0]:
        return

    if write:  
        os.makedirs(config.data.log_path, exist_ok=True)  
    log = Log(fname=config.data.log_path, write=write) 
    # PTitle(log,config.local_rank)  
    PSetting(log, 'Data Settings', config.data.keys(), config.data.values(), newline=2,
             rank=config.local_rank) 
    PSetting(log, 'Hyper Parameters', config.parameters.keys(), config.parameters.values(), rank=config.local_rank)
    PSetting(log, 'Training Settings', config.train.keys(), config.train.values(), rank=config.local_rank)
    PSetting(log, 'Other Settings', config.misc.keys(), config.misc.values(), rank=config.local_rank)

    return log


def SetupDevice():  
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ: 
        rank = int(os.environ["RANK"])  
        world_size = int(os.environ['WORLD_SIZE'])  
        torch.cuda.set_device(rank)
        # torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size,
        #                                      rank=rank)  # windows
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size,
                                             rank=rank)  # linux
        torch.distributed.barrier() 
    else:  # local
        rank = -1
        world_size = -1
    nprocess = torch.cuda.device_count()  
    torch.cuda.set_device(rank)
    # torch.backends.cudnn.benchmark = True
    return nprocess, rank


def SetSeed(config): 
    seed = config.misc.seed + config.local_rank
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    np.random.seed(seed) 
    random.seed(seed)


def ScaleLr(config):
    base_lr = config.train.lr * config.data.batch_size * get_world_size() / 512.0   # 512.0
    return base_lr


def LocateDatasets(config): 
    # config = None
    def HostIp():
        """ return: ip"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80)) 
            ip = s.getsockname()[0]  
        finally:
            s.close() 
        return ip

    ip = HostIp()
    print(ip)
    address = ip.split('.')[3]  
    data_root = config.data.data_root  
    batch_size = config.data.batch_size
    if address == 'XXX':
        data_root = '/DATA/XXX/dataset'
        batch_size = config.data.batch_size
    elif address == 'XXX':
        data_root = '/DATA/XXX/dataset'
        batch_size = config.data.batch_size
    elif address == 'XXX':
        data_root = 'F:\\Datasets' 
        batch_size = config.data.batch_size
    elif address == '118':
        data_root = '/home/XXX/dataset/' 
        batch_size = config.data.batch_size
    return data_root, batch_size


if __name__ == '__main__':
    LocateDatasets(config)
    nprocess, rank = SetupDevice()
    print(rank)
    print(type(rank))
