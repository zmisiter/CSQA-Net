import os

from settings.defaults import _C
from settings.setup_functions import *
import argparse


config = _C.clone()
cfg_file = os.path.join('configs', 'cub', 'cub_vit.yaml')
config = SetupConfig(config, cfg_file)
config.defrost()

# Log Name and Perferences
# config.write = True
# config.train.checkpoint = True  
config.misc.exp_name = f'{config.data.dataset + config.model.type}' 
config.misc.log_name = f'XXX' 
# config.cuda_visible = 'XXX' 
config.cuda_visible = 'X'  

# Environment Settings
config.data.log_path = os.path.join(config.misc.output, config.misc.exp_name, config.misc.log_name
                                    + time.strftime(' %m-%d_%H-%M', time.localtime()))

config.model.pretrained = os.path.join(config.model.pretrained,    # pretrained root
                                       config.model.name + config.model.pre_version + config.model.pre_suffix)
os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible
os.environ['OMP_NUM_THREADS'] = '1'

# Setup Functions
config.nprocess, config.local_rank = SetupDevice()
config.data.data_root, config.data.batch_size = LocateDatasets(config)  
config.train.lr = ScaleLr(config) 
log = SetupLogs(config, config.local_rank)  
if config.write and config.local_rank in [-1, 0]: 
    with open(config.data.log_path + '/config.json', "w") as f:  
        f.write(config.dump())  
config.freeze()
SetSeed(config)


# if __name__ == '__main__':
#     print(config.local_rank)
