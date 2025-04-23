import math
from functools import partial

import torch
import torch.nn.functional as F
# from timm.models import create_attn
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import DropBlock2d, DropPath
# from timm.models.layers.classifier import create_classifier
from timm.models.resnet import create_aa
from torch import nn
from einops import rearrange


# timm.models里有很多模型

def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2  # 为了使得下采样时保持原图像的大小不变被2整除
    return padding


class BasicBlock(nn.Module):  # 双层残差模块，18层和34层，但是由于是2个3×3卷积，训练参数量过大，https://zhuanlan.zhihu.com/p/475489313
    expansion = 1  # 主分支中每个stage中卷积核的个数有没有发生变化

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            # downsample对应的是虚线，即fm尺寸是否改变
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'  'ResNext中的分支条数也即分组数，basicblock只支持1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation  # 默认first_dilation=None，即当作bool类型False，与dilation作or运算 返回dilation的值
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)  # 判断and前后的逻辑表达式的值

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)  # dilation = 1等同于没有dilation的标准卷积；dilation的作用是在不同点之间有一个1的差距。
        self.bn1 = norm_layer(first_planes)  # 使用BN时不需要bias
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()  # 非空就用drop_block，空则用nn.Identity()，后者网络层的设计是用于占位的，即不干活，只是有这么一个层
        self.act1 = act_layer(inplace=True)  # inplace=True 可以节省显存，同时还可省去反复申请和释放内存的时间，但是会对原变量覆盖
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)  # aa和se都是给其他改进网络用的

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        #		self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path


class Bottleneck(nn.Module):  # 三层残差模块，50层和101层，1×1，3×3，1×1
    expansion = 4  # 残差结构所使用卷积核的变化，也就是输出通道数增加了4倍

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            aa_layer=None, drop_block=None, drop_path=None):  # 主干网络有drop_path
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)  # math.floor将输入的数字向下舍入到最接近的整数
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation  # 此时first_dilation=None，即当作bool类型False，与dilation作or运算 返回dilation的值
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,  # 因为三层残差模块中，第二个conv的步长可能为2，这里会变
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = nn.Identity()

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):  # 第一个参数永远是self，归零初始化每个残差分支中的最后一个BN，每个残差分支都从0开始，所以每个残差模块像恒等映射
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):  # 第一个参数永远是self
        shortcut = x  # 先把输入赋值给残差边

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:  # 残差边是否有卷积，即残差边是否需要进行下采样(虚线的情况)；
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


# 和普通函数相比，在类中定义函数只有一点不同，就是第一参数永远是类的本身实例变量self，并且调用时，不用传递该参数。
# 除此之外，类的方法(函数）和普通函数没啥区别，你既可以用默认参数、可变参数或者关键字参数（*args是可变参数，
# args接收的是一个tuple，**kw是关键字参数，kw接收的是一个dict）
# https://blog.csdn.net/CLHugh/article/details/75000104

def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d  # 若未指定norm，则使用bn
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        pool = nn.AvgPool2d(2, avg_stride, ceil_mode=True, count_include_pad=False)
    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob=0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]


# make_blocks做的事情就是把四个stage以及每个stage里的block实例化好
# block_fn代表是二层残差还是三层残差，他是类的实例化


def make_blocks(  # block_repeats每个block重复几次
        block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []  # stages列表
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1

    # 对channels[64，128，256，512]和block_repeats[3，4，6，3]两个列表进行遍历
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it；这里提出layer1，2，3，4
        stride = 1 if stage_idx == 0 else 2  # 第一个stage的第一层卷积前有maxpool不需要下采样，第二stage及之后的开始的卷积层需要下采样
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:  # 由于从stage2开始，每个stage开头都会做步长为2的卷积，所以if语句均成立
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(
                **down_kwargs)  # stride=2，则实例化downsample层进行下采样

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []  # blocks列表
        for block_idx in range(num_blocks):  # 每个stage中的多个block进行循环，num_blocks即循环次数
            downsample = downsample if block_idx == 0 else None  # 对于每个block的第一层卷积需要用到downsample层，其他不用
            stride = stride if block_idx == 0 else 1  # 第一个block前有maxpool不需要下采样，后面的block不需要下采样
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(  # block_fn是类的实例化，添加到blocks列表之中
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(
            *blocks)))  # *是把bloacks展开作为sequential的输入，它会把blocks进行连接构成一个stage，并赋名字为stage_name，将这个元组添加到stages列表中，返回到resnet类中去
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class ResNet(nn.Module):  # 整个网络框架

    def __init__(  # block对应的就是残差结构，会根据我们所定义的层结构传入不同的block，18，34-basic；layers是一个列表参数
            self, block, layers, num_classes=1000, in_chans=3, output_stride=32, global_pool='avg',
            cardinality=1, base_width=64, block_reduce_first=1, down_kernel_size=1, avg_down=False, act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d, aa_layer=None,
            drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., zero_init_last=True, block_args=None):
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes  # 传入到类变量中
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        inplanes = 64
        self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        # 无论哪一种resnet，除了公共部分(conv1)外，都是由4大块组成(con2_x,con3_x,con4_x,con5_x,)，
        # 每一块的起始通道数都是64，128，256，512，这点非常重要。暂且称它为“基准 通道数”

        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(  # stage_modules是由元组构成的列表，每个元组表示stage_name和stage_module对象
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        # 通过add_module函数将其添加到当前的模型作为一个子module，add_module接受两个参数，第一个是module_name，第二个是module这个对象
        # https://pytorch.org/docs/1.2.0/nn.html#torch.nn.Module.add_module
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc；
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier) 此时新模型不用他的分类器
        # self.num_features = 512 * block.expansion  得到最后输出的通道数
        # self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        # self.init_weights(zero_init_last=zero_init_last)
        # self.feature = nn.Conv2d(self.num_features, 1, 1)
        # nn.init.constant_(self.feature.weight, 0)
        # nn.init.constant_(self.feature.bias, 0)

        # layer 4 Feature map
        self.pred_map = None

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):  # 函数用于判断对象是否包含对应的属性，返回bool
                    m.zero_init_last()

    def reset_classifier(self, num_classes, global_pool='avg'):  # 分类器
        self.num_classes = num_classes

    # self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    # ***** Change *****
    # def forward_features(self, x):
    def forward(self, x):
        x = self.conv1(x)  # 输出 shape[2 64 224 224]
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)  # shape[2 64 112 112]

        # x = self.maxpool(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # 不管什么主干网络，只要是基于层次化设计的，每个阶段出来的特征图尺寸都应该是一样的
        x1 = self.layer1(x)  # 第一阶段特征图  [2, 256, 112, 112]
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)  # [2, 2048, 14, 14]
        # print(x.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # res = [x, x1, x2, x3, x4]
        # for a in res:
        # 	print(a.shape)

        return [x1, x2, x3, x4]

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)  # 对空间维度进行池化
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)  # 是否经过fc层

    # ***** Change *****
    def forward_old(self, x):
        x = self.forward_features(x)
        h, w = x.shape[-2], x.shape[-1]
        features = self.feature(x).reshape(-1, h, w)
        self.pred_map = features
        # x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        out = self.forward_head(x)
        # return out,features
        return out

    def get_pred_map(self):
        return self.pred_map


def resnet_backbone(**kwargs):  # 传递关键字参数给函数
    """Constructs a ResNet-50 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return ResNet(**model_args)


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return ResNet(**model_args)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)  # dict为内置函数，使用关键字参数创建字典
    return ResNet(**model_args)


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return ResNet(**model_args)


if __name__ == '__main__':
    x = torch.rand(2, 3, 448, 448)
    model = resnet_backbone(num_classes=200)
    y = model(x)
    for i in range(4):
        print(y[i].shape)
# print(y)
