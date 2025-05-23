import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from timm.models.helpers import named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_


class Attention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
		super().__init__()
		assert dim % num_heads == 0, 'dim should be divisible by num_heads'
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class LayerScale(nn.Module):
	def __init__(self, dim, init_values=1e-5, inplace=False):
		super().__init__()
		self.inplace = inplace
		self.gamma = nn.Parameter(init_values * torch.ones(dim))

	def forward(self, x):
		return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

	def __init__(
			self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
			drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
		self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
		# NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
		self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

		self.norm2 = norm_layer(dim)
		self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
		self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
		self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

	def forward(self, x):
		x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
		x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
		return x


class VisionTransformer(nn.Module):
	""" Vision Transformer

	A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
		- https://arxiv.org/abs/2010.11929
	"""

	def __init__(
			self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
			embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
			class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
			weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, num_cls_layers_to_output=3):
		"""
		Args:
			img_size (int, tuple): input image size
			patch_size (int, tuple): patch size
			in_chans (int): number of input channels
			num_classes (int): number of classes for classification head
			global_pool (str): type of global pooling for final sequence (default: 'token')
			embed_dim (int): embedding dimension
			depth (int): depth of transformer
			num_heads (int): number of attention heads
			mlp_ratio (int): ratio of mlp hidden dim to embedding dim
			qkv_bias (bool): enable bias for qkv if True
			init_values: (float): layer-scale init values
			class_token (bool): use class token
			fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
			drop_rate (float): dropout rate
			attn_drop_rate (float): attention dropout rate
			drop_path_rate (float): stochastic depth rate
			weight_init (str): weight init scheme
			embed_layer (nn.Module): patch embedding layer
			norm_layer: (nn.Module): normalization layer
			act_layer: (nn.Module): MLP activation layer
		"""
		super().__init__()
		assert global_pool in ('', 'avg', 'token')
		assert class_token or global_pool != 'token'
		use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
		norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
		act_layer = act_layer or nn.GELU

		self.num_classes = num_classes
		self.global_pool = global_pool
		self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
		self.num_prefix_tokens = 1 if class_token else 0
		self.no_embed_class = no_embed_class

		self.patch_embed = embed_layer(
			img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
		num_patches = self.patch_embed.num_patches

		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
		embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
		self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
		self.pos_drop = nn.Dropout(p=drop_rate)

		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
		self.blocks = nn.Sequential(*[
			block_fn(
				dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
				drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
			for i in range(depth)])
		self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
		self.num_cls_layers_to_output = num_cls_layers_to_output

		# Classifier Head
		self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
		# self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

		if weight_init != 'skip':
			self.init_weights(weight_init)

	def init_weights(self, mode=''):
		assert mode in ('jax', 'jax_nlhb', 'moco', '')
		head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
		trunc_normal_(self.pos_embed, std=.02)
		if self.cls_token is not None:
			nn.init.normal_(self.cls_token, std=1e-6)
		named_apply(get_init_weights_vit(mode, head_bias), self)

	def _init_weights(self, m):
		# this fn left here for compat with downstream users
		init_weights_vit_timm(m)

	@torch.jit.ignore()
	def load_pretrained(self, checkpoint_path, prefix=''):
		_load_weights(self, checkpoint_path, prefix)

	@torch.jit.ignore
	def no_weight_decay(self):
		return {'pos_embed', 'cls_token', 'dist_token'}

	@torch.jit.ignore
	def group_matcher(self, coarse=False):
		return dict(
			stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
			blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
		)

	@torch.jit.ignore
	def set_grad_checkpointing(self, enable=True):
		self.grad_checkpointing = enable

	@torch.jit.ignore
	def get_classifier(self):
		return self.head

	def reset_classifier(self, num_classes: int, global_pool=None):
		self.num_classes = num_classes
		if global_pool is not None:
			assert global_pool in ('', 'avg', 'token')
			self.global_pool = global_pool
		self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

	def _pos_embed(self, x):
		if self.no_embed_class:
			# deit-3, updated JAX (big vision)
			# position embedding does not overlap with class token, add then concat
			x = x + self.pos_embed
			if self.cls_token is not None:
				x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
		else:
			# original timm, JAX, and deit vit impl
			# pos_embed has entry for class token, concat then add
			if self.cls_token is not None:
				x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
			x = x + self.pos_embed
		return self.pos_drop(x)

	def forward_features(self, x):
		cls_list = []
		x = self.patch_embed(x)
		x = self._pos_embed(x)
		for i, block in enumerate(self.blocks):
			x = block(x)
			if i >= len(self.blocks) - self.num_cls_layers_to_output:
				cls_list.append(x[:, 0])
		x = self.norm(x)
		return x, cls_list

	def forward_head(self, x, pre_logits: bool = False):
		if self.global_pool:
			x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
		x = self.fc_norm(x)
		return x if pre_logits else self.head(x)

	def forward(self, x):
		x, cls_list = self.forward_features(x)
		# x = self.forward_head(x)
		return x[:,1:], cls_list 


def init_weights_vit_timm(module: nn.Module, name: str = ''):
	""" ViT weight initialization, original timm impl (for reproducibility) """
	if isinstance(module, nn.Linear):
		trunc_normal_(module.weight, std=.02)
		if module.bias is not None:
			nn.init.zeros_(module.bias)
	elif hasattr(module, 'init_weights'):
		module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
	""" ViT weight initialization, matching JAX (Flax) impl """
	if isinstance(module, nn.Linear):
		if name.startswith('head'):
			nn.init.zeros_(module.weight)
			nn.init.constant_(module.bias, head_bias)
		else:
			nn.init.xavier_uniform_(module.weight)
			if module.bias is not None:
				nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
	elif isinstance(module, nn.Conv2d):
		lecun_normal_(module.weight)
		if module.bias is not None:
			nn.init.zeros_(module.bias)
	elif hasattr(module, 'init_weights'):
		module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = ''):
	""" ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
	if isinstance(module, nn.Linear):
		if 'qkv' in name:
			# treat the weights of Q, K, V separately
			val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
			nn.init.uniform_(module.weight, -val, val)
		else:
			nn.init.xavier_uniform_(module.weight)
		if module.bias is not None:
			nn.init.zeros_(module.bias)
	elif hasattr(module, 'init_weights'):
		module.init_weights()


def get_init_weights_vit(mode='jax', head_bias: float = 0.):
	if 'jax' in mode:
		return partial(init_weights_vit_jax, head_bias=head_bias)
	elif 'moco' in mode:
		return init_weights_vit_moco
	else:
		return init_weights_vit_timm


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
	""" Load weights from .npz checkpoints for official Google Brain Flax implementation
	"""
	import numpy as np

	def _n2p(w, t=True):
		if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
			w = w.flatten()
		if t:
			if w.ndim == 4:
				w = w.transpose([3, 2, 0, 1])
			elif w.ndim == 3:
				w = w.transpose([2, 0, 1])
			elif w.ndim == 2:
				w = w.transpose([1, 0])
		return torch.from_numpy(w)

	w = np.load(checkpoint_path)
	if not prefix and 'opt/target/embedding/kernel' in w:
		prefix = 'opt/target/'

	if hasattr(model.patch_embed, 'backbone'):
		# hybrid
		backbone = model.patch_embed.backbone
		stem_only = not hasattr(backbone, 'stem')
		stem = backbone if stem_only else backbone.stem
		stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
		stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
		stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
		if not stem_only:
			for i, stage in enumerate(backbone.stages):
				for j, block in enumerate(stage.blocks):
					bp = f'{prefix}block{i + 1}/unit{j + 1}/'
					for r in range(3):
						getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
						getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
						getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
					if block.downsample is not None:
						block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
						block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
						block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
		embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
	else:
		embed_conv_w = adapt_input_conv(
			model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
	model.patch_embed.proj.weight.copy_(embed_conv_w)
	model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
	if model.global_pool == 'token':
		model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
	pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
	if pos_embed_w.shape != model.pos_embed.shape:
		pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
			pos_embed_w,
			model.pos_embed,
			getattr(model, 'num_prefix_tokens', 0),
			model.patch_embed.grid_size
		)
	model.pos_embed.copy_(pos_embed_w)
	model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
	model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
	# if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
	# 	model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
	# 	model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
	# NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
	# if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
	#     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
	#     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
	for i, block in enumerate(model.blocks.children()):
		block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
		mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
		block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
		block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
		block.attn.qkv.weight.copy_(torch.cat([
			_n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
		block.attn.qkv.bias.copy_(torch.cat([
			_n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
		block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
		block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
		for r in range(2):
			getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
			getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
		block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
		block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=()):
	# Rescale the grid of position embeddings when loading from state_dict. Adapted from
	# https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
	print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
	ntok_new = posemb_new.shape[1]
	if num_prefix_tokens:
		posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
		ntok_new -= num_prefix_tokens
	else:
		posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
	gs_old = int(math.sqrt(len(posemb_grid)))
	if not len(gs_new):  # backwards compatibility
		gs_new = [int(math.sqrt(ntok_new))] * 2
	assert len(gs_new) >= 2
	print('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
	posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
	posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
	posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
	posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
	return posemb


def vit_backbone(**kwargs):
	""" ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
	ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
	NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
	"""
	model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
	                    img_size=448,**kwargs)
	model = VisionTransformer(**model_kwargs)
	return model


if __name__ == '__main__':
	x = torch.rand(2, 3, 448, 448)
	model = vit_backbone(num_classes=200)
	y, cls_list = model(x)
	for i, block in enumerate(cls_list):
		print(y.shape)
	# torch.Size([2, 784, 768])   B N C
