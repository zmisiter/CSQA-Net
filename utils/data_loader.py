# import ml_collections
from timm.data import Mixup
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from settings.setup_functions import get_world_size
# from dataset import *
from .dataset import *
import sys


def build_transforms(config): 
    resize = config.data.resize
    normalized_info = normalized()
    if config.data.no_crop:  
        train_base = [transforms.Resize((config.data.img_size, config.data.img_size), InterpolationMode.BICUBIC),
                      transforms.RandomHorizontalFlip()] 
        test_base = [transforms.Resize((config.data.img_size, config.data.img_size), InterpolationMode.BICUBIC)]
    else: 
        train_base = [transforms.Resize((resize, resize), InterpolationMode.BICUBIC),
                      transforms.RandomCrop(config.data.img_size, padding=config.data.padding), 
                      transforms.RandomHorizontalFlip()]
        test_base = [transforms.Resize((resize, resize), InterpolationMode.BICUBIC),
                     transforms.CenterCrop(config.data.img_size)] 
    to_tensor = [transforms.ToTensor(),  
                 transforms.Normalize(normalized_info[config.data.dataset][:3],
                                      normalized_info[config.data.dataset][3:])]

    if config.data.blur > 0:  
        train_base += [
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=config.data.blur),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=config.data.blur)] 
    if config.data.color > 0:  
        train_base += [transforms.ColorJitter(config.data.color, config.data.color, config.data.color, config.data.hue)]
    if config.data.rotate > 0:
        train_base += [transforms.RandomRotation(config.data.rotate, InterpolationMode.BICUBIC)]
    if config.data.autoaug:  
        train_base += [transforms.AutoAugment(interpolation=InterpolationMode.BICUBIC)]
    train_transform = transforms.Compose([*train_base, *to_tensor])  
    test_transform = transforms.Compose([*test_base, *to_tensor]) 
    return train_transform, test_transform


def build_loader(config):  
    train_transform, test_transform = build_transforms(config)
    train_set, test_set, num_classes = None, None, None
    if config.data.dataset == 'cub':
        root = os.path.join(config.data.data_root,
                            'CUB_200_2011') 
        train_set = CUB(root, True, train_transform)  
        test_set = CUB(root, False, test_transform)  
        num_classes = 200  

    elif config.data.dataset == 'cars':
        root = os.path.join(config.data.data_root, 'cars')
        train_set = Cars(root, True, train_transform)  
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

    elif config.data.dataset == 'flowers': 
        root = os.path.join(config.data.data_root, 'flowers')
        train_set = OxfordFlowers(root, True, train_transform)
        test_set = OxfordFlowers(root, False, test_transform)
        num_classes = 102

    if config.local_rank == -1: 
        train_sampler = RandomSampler(train_set)
        test_sampler = SequentialSampler(test_set)  
    else:  
        train_sampler = DistributedSampler(train_set, num_replicas=get_world_size(),  
                                           rank=config.local_rank,
                                           shuffle=True) 
        test_sampler = DistributedSampler(test_set)
    num_workers = 0 if sys.platform == 'win32' else 16  
    # print(sys.platform)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=config.data.batch_size,
                              num_workers=num_workers, drop_last=False,  
                              pin_memory=True) 
    test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=config.data.batch_size,
                             num_workers=num_workers, shuffle=False, drop_last=False,
                             pin_memory=True) 

    mixup_fn = None
    mixup_active = config.data.mixup > 0. or config.data.cutmix > 0.
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.data.mixup, cutmix_alpha=config.data.cutmix,
            label_smoothing=config.model.label_smooth, num_classes=num_classes) 

    return train_loader, test_loader, num_classes, len(train_set), len(test_set), mixup_fn



def normalized(): 
    normalized_info = dict()  
    normalized_info['cub'] = (0.485, 0.456, 0.406, 0.229, 0.224, 0.225)  
    normalized_info['nabirds'] = (0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    normalized_info['cars'] = (0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    normalized_info['air'] = (0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    return normalized_info

if __name__ == '__main__':
	config = ml_collections.ConfigDict()
	config.data =    ml_collections.ConfigDict()
	config.data.dataset = 'dogs'
	config.data.data_root = '/data/datasets/fine-grained'
	config.data.img_size = 448
	config.data.blur = True
	config.data.local_rank = -1
	config.data.batch_size = 8
	config.data.color = 0.1
	config.local_rank = -1
	a, b, c, d, e, f = build_loader(config)
	print(c)
