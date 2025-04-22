from yacs.config import CfgNode as CN

_C = CN()
# 能传参的都传参，不要在类中写self.dropout=0.5,
# -----------------------------------------------------------------------------
# Data Settings
# -----------------------------------------------------------------------------
_C.data = CN()
_C.data.dataset = 'cub'   # 把字符串作为文件名，每个实验要自己命名；与setup中相对应
_C.data.batch_size = 8
_C.data.data_root = '/data/datasets/fine-grained'  # 这里对应的是setup_functions里面在实验室的显卡上跑的数据集位置
_C.data.img_size = 384
_C.data.resize = int(_C.data.img_size/0.75)   # random crop之前要随机放大，这是放大的尺寸
_C.data.padding = 0
_C.data.no_crop = False   # False
_C.data.autoaug = False
_C.data.blur = 0.  # 0.1
_C.data.color = 0.  # 0.2
_C.data.hue = 0.    # 0.4
_C.data.mixup = 0.  # 0.8
_C.data.cutmix = 0.  # 1.0
_C.data.rotate = 0.
_C.data.topn = 4

# -----------------------------------------------------------------------------
# Model Settings(与主干网络相关,与自己提出的方法关系不大）
# -----------------------------------------------------------------------------
_C.model = CN()
_C.model.type = 'vit'
_C.model.name = 'vit'
_C.model.baseline_model = True
_C.model.pretrained = 'pretrained'   # 本应该是路径名”pretrained.root“，不应该写成文件名，因为只有res50
_C.model.pre_version = ''    # sam_ViT-B_16.npz
_C.model.pre_suffix = '.npz'   # '.pth'
# _C.model.mps_pretrained = 'pretrained/Swin Base.pth'
_C.model.resume = ''
_C.model.num_classes = 1000
_C.model.drop_path = 0.1
_C.model.dropout = 0.0
_C.model.label_smooth = 0.1
_C.model.parameters = 0.
_C.model.no_elp = False
_C.model.no_class = False

# p2pyaml会和defaults中的值合并，覆盖掉defaults中的值，然后值会传到build里面
# -----------------------------------------------------------------------------
# Parameters Settings(能传参数的，都传参数，不要直接在代码里面写：××× self.dropout=0.5
# -----------------------------------------------------------------------------
_C.parameters = CN()
# _C.parameters.parts_ratio = 4
_C.parameters.drop = 0.

# -----------------------------------------------------------------------------
# Training Settings
# -----------------------------------------------------------------------------
_C.train = CN()
_C.train.start_epoch = 0
_C.train.epochs = 50
_C.train.warmup_epochs = 5
_C.train.weight_decay = 0.
_C.train.clip_grad = 5.0
_C.train.checkpoint = False
_C.train.lr = 2e-02
_C.train.scheduler = 'cosine'  # cosine
_C.train.lr_epoch_update = False
_C.train.optimizer = 'sgd'   # sgd
_C.train.freeze_backbone = False
_C.train.eps = 1e-8
_C.train.betas = (0.9, 0.999)
_C.train.momentum = 0.9

# -----------------------------------------------------------------------------
# Elp Settings
# -----------------------------------------------------------------------------
_C.elp = CN()
_C.elp.init = 2
_C.elp.alpha = 0.5
_C.elp.gamma = 2
_C.elp.lr = 0.01
# -----------------------------------------------------------------------------
# Misc Settings
# -----------------------------------------------------------------------------
_C.misc = CN()
_C.misc.amp = True
_C.misc.output = './output'
_C.misc.exp_name = _C.data.dataset   # 与setup里面对应，把字符串作为文件名，每个实验要自己命名
_C.misc.log_name = 'base'
_C.data.log_path = ''
_C.misc.eval_every = 1
_C.misc.seed = 3407
_C.misc.eval_mode = False   # False
_C.misc.throughput = False
_C.misc.fused_window = False

_C.write = False
_C.local_rank = -1
_C.device = 'cuda'
_C.cuda_visible = '0,1'    # 把其他几块别人正在用的屏蔽掉，只留下没有用到的编号；这里每次是需要改的，程序在调用的时候会重新给显卡赋予一个编号
# -----------------------------------------------------------------------------
