# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

from functools import partial
import torch.nn as nn
from timm.models.registry import register_model

from neural_transformer import NeuralTransformer


# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': .9, 'interpolation': 'bicubic',
#         'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
#         **kwargs
#     }

@register_model
def labram_base_patch200_200(pretrained=False, **kwargs) -> NeuralTransformer:
    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),  # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # model.default_cfg = _cfg()
    return model


@register_model
def labram_large_patch200_200(pretrained=False, **kwargs) -> NeuralTransformer:
    model = NeuralTransformer(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, out_chans=16,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),  # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # model.default_cfg = _cfg()
    return model


@register_model
def labram_huge_patch200_200(pretrained=False, **kwargs) -> NeuralTransformer:
    model = NeuralTransformer(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, out_chans=32,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),  # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # model.default_cfg = _cfg()
    return model
