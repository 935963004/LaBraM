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
import torch
import torch.nn as nn
from timm.models.registry import register_model

from models.masked_neural_transformer import NeuralTransformerForMEM


@register_model
def labram_base_patch200_1600_8k_vocab(pretrained=False, **kwargs) -> NeuralTransformerForMEM:  # 5M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = NeuralTransformerForMEM(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    # model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def labram_large_patch200_1600_8k_vocab(pretrained=False, **kwargs) -> NeuralTransformerForMEM:  # 50M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = NeuralTransformerForMEM(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    # model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def labram_huge_patch200_1600_8k_vocab(pretrained=False, **kwargs) -> NeuralTransformerForMEM:  # 380M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = NeuralTransformerForMEM(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=32,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    # model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
