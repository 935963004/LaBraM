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
from timm.models.registry import register_model
from torch import nn

from models.vqnsp import VQNSP


@register_model
def vqnsp_encoder_base_decoder_3x200x12(pretrained=False, pretrained_weight=None, as_tokenzer=False, EEG_size=1600,
                                        n_code=8192, code_dim=32, **kwargs) -> VQNSP:
    encoder_config, decoder_config = get_vqnsp_model_default_params(), get_vqnsp_model_default_params()

    # encoder settings
    encoder_config['EEG_size'] = EEG_size
    encoder_config['num_classes'] = 0
    # encoder_config['use_mean_pooling'] = False
    # decoder settings
    decoder_config['EEG_size'] = EEG_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = code_dim
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 3
    decoder_out_dim = 200

    model = VQNSP(encoder_config, decoder_config, n_code, code_dim,
                  decoder_out_dim=decoder_out_dim, **kwargs)

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None
        model.encoder.use_mean_pooling = False  # TODO: must be changed in config
        model.decoder.use_mean_pooling = False  # TODO: must be changed in config
        if pretrained_weight.startswith('https'):
            weights = torch.hub.load_state_dict_from_url(pretrained_weight, map_location='cpu', check_hash=True)
        else:
            weights = torch.load(pretrained_weight, map_location='cpu', weights_only=False)

        if 'model' in weights:
            weights = weights['model']
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())

        for k in keys:
            if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
                del weights[k]

        weights = {k.replace(".fc_norm.", ".norm."): v for k, v in weights.items()}
        model.load_state_dict(weights)
    return model


@register_model
def vqnsp_encoder_large_decoder_3x200x24(pretrained=False, pretrained_weight=None, as_tokenzer=False, EEG_size=1600,
                                         n_code=8192, code_dim=32, **kwargs) -> VQNSP:
    encoder_config, decoder_config = get_vqnsp_model_default_params(), get_vqnsp_model_default_params()

    # encoder settings
    encoder_config['EEG_size'] = EEG_size
    encoder_config['num_classes'] = 0
    encoder_config['depth'] = 24
    # encoder_config['use_mean_pooling'] = False
    # decoder settings
    decoder_config['EEG_size'] = EEG_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = code_dim
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 3
    decoder_out_dim = 200

    model = VQNSP(encoder_config, decoder_config, n_code, code_dim,
                  decoder_out_dim=decoder_out_dim, **kwargs)

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None
        model.encoder.use_mean_pooling = False  # TODO: must be changed in config
        model.decoder.use_mean_pooling = False  # TODO: must be changed in config
        if pretrained_weight.startswith('https'):
            weights = torch.hub.load_state_dict_from_url(pretrained_weight, map_location='cpu', check_hash=True)
        else:
            weights = torch.load(pretrained_weight, map_location='cpu')

        if 'model' in weights:
            weights = weights['model']
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())

        for k in keys:
            if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
                del weights[k]

        weights = {k.replace(".fc_norm.", ".norm."): v for k, v in weights.items()}

        model.load_state_dict(weights)
    return model


def get_vqnsp_model_default_params() -> dict:
    return dict(EEG_size=1600, patch_size=200, in_chans=1, num_classes=1000, embed_dim=200, depth=12, num_heads=10,
                mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., use_abs_pos_emb=True,
                use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)


if __name__ == '__main__':
    pass
