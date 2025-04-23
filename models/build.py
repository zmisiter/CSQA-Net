import os
import timm
import torch.nn.init

from models.backbone.ResNet import *
from models.backbone.Swin_Transformer import swin_backbone, swin_backbone_tiny
from models.backbone.ViT import vit_backbone
from models.model import CurriculumConv


# 想用其他主干网络的时候，把标准的copy到一个新文件里面，然后再把需要的函数拷进去，然后调用这个函数就好了

def build_models(config, num_classes):
    if config.model.baseline_model:  # 调用defaults中的参数，默认baseline是false;
        model = baseline_models(config, num_classes)  # 对于baseline和backbone带不带分类头的概念-->看函数调用
        load_pretrained(config, model)
        return model  # return返回之后，后边就不执行了；所以上面是带头的，下面是不带头的
    dim, backbone = 512, None
    if config.model.type.lower() == 'resnet':  # .lower()函数是不管大写小写，都变成小写
        dim = 2048
        if config.model.name.lower() == 'resnet-50':
            backbone = resnet50()  # 魔改的res50没有分类头；这里可以加drop_path_rate=config.model.drop_path
        elif config.model.name.lower() == 'resnet-101':
            backbone = resnet101()
        elif config.model.name.lower() == 'resnet-34':
            backbone = resnet34()
        else:
            backbone = resnet_backbone(num_classes=num_classes)

    elif config.model.type.lower() == 'swin':
        if config.model.name.lower() == 'swin tiny':
            dim = 768
            backbone = swin_backbone_tiny(num_classes=num_classes, drop_path_rate=config.model.drop_path,
                                          img_size=config.data.img_size, window_size=config.data.img_size // 32)
        else:
            dim = 1024
            backbone = swin_backbone(num_classes=num_classes, drop_path_rate=config.model.drop_path,
                                     img_size=config.data.img_size, window_size=config.data.img_size // 32)

    elif config.model.type.lower() == 'vit':
        dim = 768
        backbone = vit_backbone(num_classes=num_classes)

    load_pretrained(config, backbone)
    basic_dim = dim // 4  # 原本是512，其实起名字不规范，应该是瓶颈更好；四阶段要改成256  resnet 是 dim //4
    model = CurriculumConv(backbone, basic_dim, dim, num_classes, config.data.topn, config.data.img_size//32,
                           config.model.label_smooth, config.elp.init, config.elp.alpha, config.elp.gamma,
                           config.data.img_size//2)
    return model  # 这里就是defaults中说的链路，config.parameters.drop


def baseline_models(config, num_classes):  # build_models里面backbone去掉了分类头，这里使用原来的分类头，另外输入需要的类别数
    model = None
    type = config.model.type.lower()
    if type == 'resnet':
        model = timm.models.create_model('resnet50', pretrained=False,
                                         # drop_path_rate=config.model.drop_path,
                                         num_classes=num_classes)  # 这里是带分类头的，类别数是自己输入的

    elif type == 'vit':
        model = timm.models.create_model('vit_base_patch16_224_in21k', pretrained=False,
                                         num_classes=num_classes, img_size=config.data.img_size)
    elif type == 'swin':
        if config.model.name.lower() == 'swin tiny':
            model = timm.models.create_model('swin_tiny_patch4_window7_224', pretrained=False,
                                             num_classes=num_classes, drop_path_rate=config.model.drop_path,
                                             img_size=config.data.img_size)
        else:
            model = timm.models.create_model('swin_base_patch4_window12_384_in22k', pretrained=False,
                                             num_classes=num_classes, drop_path_rate=config.model.drop_path)
    elif type == 'swinv2':
        model = timm.models.create_model('swin_large_patch4_window12_384_in22k', pretrained=False,
                                         num_classes=num_classes, drop_path_rate=config.model.drop_path)

    return model


def load_pretrained(config, model):
    if config.local_rank in [-1, 0]:
        print('-' * 11, f'Loading weight \'{config.model.pretrained}\' for fine-tuning'.center(56), '-' * 11)

    if os.path.splitext(config.model.pretrained)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(config.model.pretrained)
            if config.local_rank in [-1, 0]:
                print('-' * 18, f'Loaded successfully \'{config.model.pretrained}\''.center(42), '-' * 18)

            torch.cuda.empty_cache()
            return

    checkpoint = torch.load(config.model.pretrained, map_location='cpu')
    state_dict = None
    type = config.model.type.lower()

    if type == 'vit':
        state_dict = checkpoint
        del state_dict['head.fc.weight']
        del state_dict['head.fc.bias']
        if config.model.baseline_model:
            torch.nn.init.constant_(model.head.fc.bias, 0.)
            torch.nn.init.constant_(model.head.fc.weight, 0.)
        relative_position_index_keys = [k for k in state_dict.keys() if "rel_pos" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

    if type == 'resnet':
        try:
            state_dict = checkpoint['state_dict']
        except:
            state_dict = checkpoint
        # print(state_dict.keys())
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        if config.model.baseline_model:
            torch.nn.init.constant_(model.fc.bias, 0.)
            torch.nn.init.constant_(model.fc.weight, 0.)

    # fc_pretrained = state_dict['fc.bias']
    # Nc1 = fc_pretrained.shape[0]
    # Nc2 = model.fc.bias.shape[0]
    # if Nc1!=Nc2:
    # 	torch.nn.init.constant_(model.fc.bias, 0.)
    # 	torch.nn.init.constant_(model.fc.weight, 0.)
    # 	del state_dict['fc.weight']
    # 	del state_dict['fc.bias']

    elif type == 'swin' or type == 'swinv2':
        state_dict = checkpoint['model']
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete relative_coords_table since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        # Modify Patch_Merging
        if not config.model.baseline_model:
            patch_merging_keys = [k for k in state_dict.keys() if "downsample" in k]
            patch_merging_pretrained = []
            new_keys = []
            for k in patch_merging_keys:
                patch_merging_pretrained.append(state_dict[k])
                del state_dict[k]
                k = k.replace(k[7], f'{int(k[7]) + 1}')
                new_keys.append(k)

            for nk, nv in zip(new_keys, patch_merging_pretrained):
                state_dict[nk] = nv
        # print(patch_merging)

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        # relative_position_bias_table_keys = [x for x in relative_position_bias_table_keys if 'layers.3.' not in x]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = model.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()

            if nH1 != nH2:
                print(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    # bicubic interpolate relative_position_bias_table if not match
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                        mode='bicubic')

                    state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
        # bicubic interpolate absolute_pos_embed if not match
        absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
        for k in absolute_pos_embed_keys:
            # dpe
            absolute_pos_embed_pretrained = state_dict[k]
            absolute_pos_embed_current = model.state_dict()[k]
            _, L1, C1 = absolute_pos_embed_pretrained.size()
            _, L2, C2 = absolute_pos_embed_current.size()
            if C1 != C1:
                print(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                    absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                        absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                    state_dict[k] = absolute_pos_embed_pretrained_resized

        # # check classifier, if not match, then re-init classifier to zero
        # head_bias_pretrained = state_dict['head.bias']
        # Nc1 = head_bias_pretrained.shape[0]
        # Nc2 = model.head.bias.shape[0]
        # if (Nc1 != Nc2):
        if config.model.baseline_model:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
        del state_dict['head.weight']
        del state_dict['head.bias']

    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)
    if config.local_rank in [-1, 0]:
        print('-' * 16, ' Loaded successfully \'{:^22}\' '.format(config.model.pretrained), '-' * 16)

    del checkpoint
    torch.cuda.empty_cache()

# 把主干网络参数冻结，只训练自己加的东西，主干网络保持不变；反向传播导致训练速度很慢
# 假设把整个网络的参数全部锁住，则速度和推理(推理不需要反向)速度一样快，因为不需要传播baseline的参数

def freeze_backbone(model, freeze_params=False):
    if freeze_params:  # 只有freeze_params为真时才冻结，锁起来/不锁起来都跑一下
        for name, parameter in model.named_parameters():
            if name.startswith('backbone'):
                parameter.requires_grad = False


if __name__ == '__main__':
    model = build_models(1, 200)
    print(model)
