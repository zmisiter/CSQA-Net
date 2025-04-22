# import ml_collections
from timm.data import Mixup
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from settings.setup_functions import get_world_size
# from dataset import *
from .dataset import *
import sys


def build_transforms(config):  # 可能需要自己改，加载的一些数据增强
    resize = config.data.resize
    normalized_info = normalized()
    if config.data.no_crop:  # 没有random_crop,直接进行缩放
        train_base = [transforms.Resize((config.data.img_size, config.data.img_size), InterpolationMode.BICUBIC),
                      transforms.RandomHorizontalFlip()]  # Resize成指定HW；三次插值法；以默认p=0.5概率随机水平翻转图像
        test_base = [transforms.Resize((config.data.img_size, config.data.img_size), InterpolationMode.BICUBIC)]
    else:  # 若有随机裁剪，则Resize成resize的大小
        train_base = [transforms.Resize((resize, resize), InterpolationMode.BICUBIC),
                      transforms.RandomCrop(config.data.img_size, padding=config.data.padding),  # 随即裁剪到448
                      transforms.RandomHorizontalFlip()]
        test_base = [transforms.Resize((resize, resize), InterpolationMode.BICUBIC),
                     transforms.CenterCrop(config.data.img_size)]  # 中心裁剪到(img_size, img_size)
    to_tensor = [transforms.ToTensor(),  # 前三是每个通道的均值，后三是每个通道的方差
                 transforms.Normalize(normalized_info[config.data.dataset][:3],
                                      normalized_info[config.data.dataset][3:])]

    if config.data.blur > 0:  # 若启用模糊就改成非0，若不启用则改成0；启用的话则调用高斯模糊
        train_base += [
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=config.data.blur),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=config.data.blur)]  # sharpness是图像的清晰度；
    if config.data.color > 0:  # 颜色调整，随机的颜色抖动
        train_base += [transforms.ColorJitter(config.data.color, config.data.color, config.data.color, config.data.hue)]
    if config.data.rotate > 0:
        train_base += [transforms.RandomRotation(config.data.rotate, InterpolationMode.BICUBIC)]
    if config.data.autoaug:  # 在car数据集上比较好，在其他数据集上效果不明显
        train_base += [transforms.AutoAugment(interpolation=InterpolationMode.BICUBIC)]
    # 用*号将两个列表的元素拆开成独立的元素，然后再用compose组合多个步骤   https://blog.csdn.net/weixin_40877427/article/details/82931899
    train_transform = transforms.Compose([*train_base, *to_tensor])  # if语句还可以自行扩展，然后在defaults中添加相应的参数
    test_transform = transforms.Compose([*test_base, *to_tensor])  # 对test做归一化处理不算作弊把。。。
    return train_transform, test_transform


def build_loader(config):  # 给每个数据集作自定义，这里给每个数据集命名
    train_transform, test_transform = build_transforms(config)
    train_set, test_set, num_classes = None, None, None
    if config.data.dataset == 'cub':
        root = os.path.join(config.data.data_root,
                            'CUB_200_2011')  # 数据集目录，可能在windows或者linux用，自动判断是/还是\；windows用\，linux等用/
        train_set = CUB(root, True, train_transform)  # 训练集
        test_set = CUB(root, False, test_transform)  # 测试集
        num_classes = 200  # 类别数

    elif config.data.dataset == 'cars':
        root = os.path.join(config.data.data_root, 'cars')
        train_set = Cars(root, True, train_transform)  # (根目录，train=true,转换的操作)
        test_set = Cars(root, False, test_transform)
        num_classes = 196

    elif config.data.dataset == 'dogs':
        root = os.path.join(config.data.data_root, 'Dogs')
        train_set = Dogs(root, True, train_transform)
        test_set = Dogs(root, False, test_transform)
        num_classes = 120

    elif config.data.dataset == 'air':
        root = config.data.data_root
        train_set = Aircraft(root, True, train_transform)
        test_set = Aircraft(root, False, test_transform)
        num_classes = 100

    elif config.data.dataset == 'nabirds':
        root = os.path.join(config.data.data_root, 'nabirds')
        train_set = NABirds(root, True, train_transform)
        test_set = NABirds(root, False, test_transform)
        num_classes = 555

    elif config.data.dataset == 'pet':
        root = os.path.join(config.data.data_root, 'pets')
        train_set = OxfordIIITPet(root, True, train_transform)
        test_set = OxfordIIITPet(root, False, test_transform)
        num_classes = 37

    elif config.data.dataset == 'flowers':  # 若要加别的数据集，也是一样的格式在后面添加
        root = os.path.join(config.data.data_root, 'flowers')
        train_set = OxfordFlowers(root, True, train_transform)
        test_set = OxfordFlowers(root, False, test_transform)
        num_classes = 102

    ### 数据集的加载函数在dataset里面，直接去github上面搜，接口都是一样的；inat或者imagenet都从网上找接口

    if config.local_rank == -1:  # local_rank == -1代表不用DDP，用本地的显卡跑
        train_sampler = RandomSampler(train_set)
        # 在训练集中数据随机采样,默认replacement=False  https://blog.csdn.net/u010137742/article/details/100996937
        test_sampler = SequentialSampler(test_set)  # 顺序采样
    else:  # 如果调用了DDP，那么local_rank必然不是-1；就是0,1...
        train_sampler = DistributedSampler(train_set, num_replicas=get_world_size(),  # DDP中的sampler可以对不同进程分发数据
                                           rank=config.local_rank,
                                           shuffle=True)  # https://blog.csdn.net/shenjianhua005/article/details/121485522
        test_sampler = DistributedSampler(test_set)
    num_workers = 0 if sys.platform == 'win32' else 16  # 本地跑windows系统是0，linux系统下是16, num_workers和get_world_size不一样么？？？
    # print(sys.platform)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=config.data.batch_size,
                              # pin_memory=True https://www.zhihu.com/question/356098644
                              num_workers=num_workers, drop_last=False,  # 每个epoch都是不一样的，所以每张图片都能训练到
                              pin_memory=True)  # drop_last的含义是比如100张图片，batchsize为3，则训练时将最后一张剩的图片丢掉
    test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=config.data.batch_size,
                             num_workers=num_workers, shuffle=False, drop_last=False,
                             pin_memory=True)  # 实例化
    # https://blog.csdn.net/qq_28057379/article/details/115427052

    mixup_fn = None
    mixup_active = config.data.mixup > 0. or config.data.cutmix > 0.
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.data.mixup, cutmix_alpha=config.data.cutmix,
            # mixup：将随机的两个样本按照比例混合；cumix：将一部分区域cut掉，然后随机填充训练集中的其他数据的区域值
            label_smoothing=config.model.label_smooth, num_classes=num_classes)  # 这里也有label_smoothing

    return train_loader, test_loader, num_classes, len(train_set), len(test_set), mixup_fn


# 每个图像三个通道，先reshape成600×600，再centercrop成448：[3,448,448];img.mean(),img.std(),套几个for循环，每个数据集的训练样本都搞一遍
def normalized():  # 一般情况下每个数据集都用一个固定值；但是用某些手法将数据集跑了一遍，就变成如下了
    normalized_info = dict()  # 新建空字典
    normalized_info['pet'] = (0.4817828, 0.4497765, 0.3961324, 0.26035318, 0.25577134, 0.2635264)
    normalized_info['cub'] = (0.485, 0.456, 0.406, 0.229, 0.224, 0.225)  # imagenet的均值和方差
    # normalized_info['cub'] = (0.4865833, 0.5003001, 0.43229204, 0.22157472, 0.21690948, 0.24466534)
    # normalized_info['nabirds'] = (0.49218804, 0.50868344, 0.46445918, 0.21430683, 0.21335651, 0.25660837)
    normalized_info['nabirds'] = (0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    normalized_info['dogs'] = (0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    # normalized_info['dogs'] = (0.4764075, 0.45210016, 0.3912831, 0.256719, 0.25130147, 0.25520605)
    normalized_info['cars'] = (0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    # normalized_info['cars'] = (0.47026777, 0.45981872, 0.4548266, 0.2880712, 0.28685528, 0.29420388)
    normalized_info['air'] = (0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    # normalized_info['air'] = (0.47890663, 0.510387, 0.5342661, 0.21548453, 0.2100707, 0.24122715)
    normalized_info['flowers'] = (0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    # normalized_info['flowers'] = (0.4358411, 0.37802523, 0.28822893, 0.29247612, 0.24125645, 0.2639247)
    return normalized_info

# if __name__ == '__main__':
# 	config = ml_collections.ConfigDict()
# 	config.data =    ml_collections.ConfigDict()
# 	config.data.dataset = 'dogs'
# 	config.data.data_root = '/data/datasets/fine-grained'
# 	config.data.img_size = 448
# 	config.data.blur = True
# 	config.data.local_rank = -1
# 	config.data.batch_size = 8
# 	config.data.color = 0.1
# 	config.local_rank = -1
# 	a, b, c, d, e, f = build_loader(config)
# 	print(c)
