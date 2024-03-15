'''
# Adapted from https://github.com/baaivision/EVA/tree/master/EVA-CLIP
'''

from math import pi
import torch
from torch import nn
from einops import rearrange, repeat
import logging
from transformers import PretrainedConfig
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union



class EvaClipVisionConfig(PretrainedConfig):
    model_type = "evaclip_vision_model"
    def __init__(
            self,
            embed_dim=768,
            image_size=336,
            layers=24,
            width=1024,
            drop_path_rate=0,
            head_width=64,
            mlp_ratio=2.6667,
            patch_size=14,
            eva_model_name="eva-clip-l-14-336",
            xattn=True,
            fusedLN=True,
            rope=True,
            pt_hw_seq_len=16,
            intp_freq=True,
            qkv_bias=True,
            postnorm=False,
            naiveswiglu=True,
            subln=True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.image_size = image_size,
        self.layers = layers,
        self.width = width,
        self.drop_path_rate = drop_path_rate,
        self.head_width = head_width,
        self.mlp_ratio = mlp_ratio,
        self.patch_size = patch_size,
        self.eva_model_name = eva_model_name,
        self.xattn = xattn,
        self.fusedLN =fusedLN,
        self.rope = rope,
        self.pt_hw_seq_len = pt_hw_seq_len,
        self.intp_freq = intp_freq,
        self.qkv_bias = qkv_bias,
        self.postnorm = postnorm,
        self.naiveswiglu = naiveswiglu,
        self.subln = subln,

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig

        if config_dict.get("model_type") == "evaclip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig


    
class EvaClipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EvaClipVisionConfig
    base_model_prefix = "eva-clip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
            self,
            dim,
            pt_seq_len,
            ft_seq_len=None,
            custom_freqs=None,
            freqs_for='lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,
            patch_dropout=0.
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.patch_dropout = patch_dropout

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        logging.info(f'Shape of rope freq: {self.freqs_cos.shape}')

    def forward(self, t, patch_indices_keep=None):
        if patch_indices_keep is not None:
            batch = t.size()[0]
            batch_indices = torch.arange(batch)
            batch_indices = batch_indices[..., None]

            freqs_cos = repeat(self.freqs_cos, 'i j -> n i m j', n=t.shape[0], m=t.shape[1])
            freqs_sin = repeat(self.freqs_sin, 'i j -> n i m j', n=t.shape[0], m=t.shape[1])

            freqs_cos = freqs_cos[batch_indices, patch_indices_keep]
            freqs_cos = rearrange(freqs_cos, 'n i m j -> n m i j')
            freqs_sin = freqs_sin[batch_indices, patch_indices_keep]
            freqs_sin = rearrange(freqs_sin, 'n i m j -> n m i j')

            return t * freqs_cos + rotate_half(t) * freqs_sin

        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info(f"os.getenv('RoPE')={os.getenv('RoPE')}")

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        if self.training and os.getenv('RoPE') == '1':
            return x, patch_indices_keep

        return x


# --------------------------------------------------------
# Adapted from  https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import drop_path, to_2tuple, trunc_normal_
except:
    from timm.layers import drop_path, to_2tuple, trunc_normal_

if os.getenv('ENV_TYPE') == 'deepspeed':
    try:
        from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
    except:
        from torch.utils.checkpoint import checkpoint
else:
    from torch.utils.checkpoint import checkpoint

import xformers.ops as xops


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            drop=0.,
            subln=False,

    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, xattn=False, rope=None, subln=False,
            norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.subln = subln
        if self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        # self.proj = nn.Linear(all_head_dim, all_head_dim)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        if self.subln:
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, N, C
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        else:

            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope:
            # slightly fast impl
            q_t = q[:, :, 1:, :]
            ro_q_t = self.rope(q_t)
            q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

            k_t = k[:, :, 1:, :]
            ro_k_t = self.rope(k_t)
            k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)

        if self.xattn:
            q = q.permute(0, 2, 1, 3)  # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            x = xops.memory_efficient_attention(
                q, k, v,
                p=self.xattn_drop,
                scale=self.scale,
            )
            x = x.reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            if self.relative_position_bias_table is not None:
                relative_position_bias = \
                    self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.type_as(attn)

            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, xattn=False, rope=None, postnorm=False,
                 subln=False, naiveswiglu=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
            xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                subln=subln,
                drop=drop
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                x = x + self.drop_path(
                    self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(
                    self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww



try:
    from apex.normalization import FusedLayerNorm
except:
    FusedLayerNorm = LayerNorm
    print(
        "Please build and install Nvidia apex package with option '--cuda_ext' according to https://github.com/NVIDIA/apex#from-source .")
    
class EVAVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, config: EvaClipVisionConfig, drop_rate=0., use_abs_pos_emb=True, 
                 use_rel_pos_bias=False, use_shared_rel_pos_bias=False, attn_drop_rate=0.,
                 init_scale=0.001, grad_checkpointing=False, qk_scale=None, 
                 ):
        
        super().__init__()
        in_chans = 3
        embed_dim = config.width
        depth = config.layers
        print(config.widt)
        print(config.head_width)
        num_heads = config.width // config.head_width

        norm_layer = partial(FusedLayerNorm, eps=1e-6) if config.fusedLN else partial(LayerNorm, eps=1e-6),
        self.config = config
        self.image_size = config.image_size
        self.num_classes = config.embed_dim

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=config.image_size, patch_size=config.patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if config.rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = config.image_size // config.patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=config.pt_hw_seq_len,
                ft_seq_len=hw_seq_len if config.intp_freq else None,
                # patch_dropout=patch_dropout
            )
        else:
            self.rope = None

        self.naiveswiglu = config.naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                init_values=config.init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                xattn=config.xattn, rope=self.rope, postnorm=config.postnorm, subln=config.subln, naiveswiglu=config.naiveswiglu)
            for i in range(depth)])
        self.norm = nn.Identity() if config.use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if config.use_mean_pooling else None
        self.head = nn.Linear(embed_dim, config.embed_dim) if config.embed_dim > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(config.patch_dropout) if config.patch_dropout > 0. else nn.Identity()

        self.grad_checkpointing = grad_checkpointing

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_cast_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, return_all_features=False):

        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                continue
            if self.grad_checkpointing:
                x = checkpoint(blk, x, (rel_pos_bias,))
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return x[:, 0]
        return x

    def forward(self, x, return_all_features=False):
        if return_all_features:
            return self.forward_features(x, return_all_features)
        x = self.forward_features(x)
        x = self.head(x)
        return x


class EvaClipVisionModel(EvaClipPreTrainedModel):

    config_class = EvaClipVisionConfig

    def __init__(self, config: EvaClipVisionConfig):
        super(EvaClipVisionModel, self).__init__(config)

        self.vision_model = EVAVisionTransformer(config)

    def forward(self, image, **kwargs):
        encode = self.vision_model(image, return_all_features=True)[:, 1:, :]
        return encode

    @property
    def dtype(self):
        return list(self.parameters())[-1].dtype

    @property
    def device(self):
        return list(self.parameters())[-1].device

