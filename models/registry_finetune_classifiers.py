# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

from typing import Optional

from timm.models import register_model

from configs import ConfigNeuralTransformer
from models.neural_transformer import NeuralTransformer
from layers.norm_layers import NormLayer

def get_default_cfg_base_patch200_200(**kwargs) -> ConfigNeuralTransformer:
    return ConfigNeuralTransformer(depth=12,
                                   patch_size=200,
                                   embed_dim=200,
                                   num_heads=10,
                                   mlp_ratio=4,
                                   qk_norm=NormLayer.LAYER_NORM.value,
                                   norm_layer=NormLayer.LAYER_NORM.value,
                                   **kwargs)


def get_default_cfg_large_patch200_200(**kwargs) -> ConfigNeuralTransformer:
    return ConfigNeuralTransformer(depth=24,
                                   patch_size=200,
                                   embed_dim=400,
                                   num_heads=16,
                                   mlp_ratio=4,
                                   out_chans=16,
                                   qk_norm=NormLayer.LAYER_NORM.value,
                                   norm_layer=NormLayer.LAYER_NORM.value,
                                   **kwargs)


def get_default_cfg_huge_patch200_200(**kwargs) -> ConfigNeuralTransformer:
    return ConfigNeuralTransformer(depth=48,
                                   patch_size=200,
                                   embed_dim=800,
                                   num_heads=16,
                                   mlp_ratio=4,
                                   out_chans=32,
                                   qk_norm=NormLayer.LAYER_NORM.value,
                                   norm_layer=NormLayer.LAYER_NORM.value,
                                   )

@register_model
def labram_base_patch200_200(cfg: Optional[ConfigNeuralTransformer] = None, **kwargs) -> NeuralTransformer:
    if cfg is None:
        cfg = get_default_cfg_base_patch200_200(**kwargs)
    model = NeuralTransformer(cfg)
    # patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
    # qk_norm=partial(nn.LayerNorm, eps=1e-6),  # qkv_bias=True,
    # norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # model.default_cfg = _cfg()
    return model


@register_model
def labram_large_patch200_200(cfg: Optional[ConfigNeuralTransformer] = None, **kwargs) -> NeuralTransformer:
    if cfg is None:
        cfg = get_default_cfg_large_patch200_200(**kwargs)

    model = NeuralTransformer(cfg)
    # patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, out_chans=16,
    # qk_norm=partial(nn.LayerNorm, eps=1e-6),  # qkv_bias=True,
    # norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # model.default_cfg = _cfg()
    return model


@register_model
def labram_huge_patch200_200(cfg: Optional[ConfigNeuralTransformer] = None, **kwargs) -> NeuralTransformer:
    if cfg is None:
        cfg = get_default_cfg_huge_patch200_200(**kwargs)
    model = NeuralTransformer(cfg)
    # patch_size=200,
    # embed_dim=800,
    # depth=48, num_heads=16,
    # mlp_ratio=4,
    # out_chans=32,
    # qk_norm=partial(nn.LayerNorm, eps=1e-6),  # qkv_bias=True,
    # norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # model.default_cfg = _cfg()
    return model
