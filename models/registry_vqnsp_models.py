# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
from pathlib import Path
from typing import Optional

import torch
from timm.models import register_model

from configs import ConfigVQNSP, DEFAULT_CODEBOOK_SIZE, \
    DEFAULT_CODEBOOK_EMBED_DIM, \
    DEFAULT_CODEBOOK_DECAY
from models.vqnsp_model import VQNSP


def get_cfg_vqnsp_base_3x200x12(
        # eeg_size: int = 1600,
        decay_codebook: float = DEFAULT_CODEBOOK_DECAY,
        num_codebook_tokens: int = DEFAULT_CODEBOOK_SIZE,
        codebook_dim: int = DEFAULT_CODEBOOK_EMBED_DIM,
        quantize_kmeans_init: bool = True,
        **kwargs) -> ConfigVQNSP:
    cfg = ConfigVQNSP()
    # cfg.encoder.eeg_size = eeg_size
    cfg.encoder.num_classes = 0
    cfg.encoder.depth = 12
    cfg.decoder.num_classes = 0
    cfg.decoder.depth = 3
    cfg.decoder.in_chans = codebook_dim
    cfg.decoder.eeg_size = cfg.encoder.eeg_size // cfg.decoder.patch_size
    cfg.codebook.decay_codebook_mov_avg = decay_codebook
    cfg.codebook.num_tokens = num_codebook_tokens
    cfg.codebook.emb_dim = codebook_dim
    cfg.codebook.kmeans_init = quantize_kmeans_init
    cfg.update(kwargs)
    return cfg


def get_cfg_vqnsp_large_3x200x12(eeg_size: int = 1600,
                                 decay_codebook: float = DEFAULT_CODEBOOK_DECAY,
                                 num_codebook_tokens: int = DEFAULT_CODEBOOK_SIZE,
                                 codebook_dim: int = DEFAULT_CODEBOOK_EMBED_DIM,
                                 quantize_kmeans_init: bool = True,
                                 **kwargs
                                 ) -> ConfigVQNSP:
    cfg = ConfigVQNSP()
    cfg.encoder.eeg_size = eeg_size
    cfg.encoder.num_classes = 0
    cfg.encoder.depth = 24
    cfg.decoder.num_classes = 0
    cfg.decoder.depth = 3
    cfg.decoder.eeg_size = cfg.encoder.eeg_size // cfg.decoder.patch_size
    cfg.codebook.decay_codebook_mov_avg = decay_codebook
    cfg.codebook.num_tokens = num_codebook_tokens
    cfg.codebook.emb_dim = codebook_dim
    cfg.codebook.kmeans_init = quantize_kmeans_init
    cfg.update(kwargs)
    return cfg


@register_model
def vqnsp_encoder_base_decoder_3x200x12(cfg: Optional[ConfigVQNSP] = None,
                                        weights_path: Optional[str] = None,
                                        as_tokenizer: bool = False,
                                        # eeg_size: int = 1600,
                                        # decay_codebook: float = DEFAULT_CODEBOOK_DECAY,
                                        # num_codebook_tokens: int = DEFAULT_CODEBOOK_SIZE,
                                        # codebook_dim: int = DEFAULT_CODEBOOK_EMBED_DIM,
                                        # quantize_kmeans_init: bool =True,
                                        **kwargs) -> VQNSP:
    if cfg is None:
        cfg = get_cfg_vqnsp_base_3x200x12(**kwargs)
    # encoder_config, decoder_config = get_vqnsp_default_cfg(), get_vqnsp_default_cfg()
    # cfg = ConfigVQNSP()
    # cfg.encoder.eeg_size = eeg_size
    # cfg.encoder.num_classes = 0
    # cfg.encoder.depth = 12
    # cfg.decoder.num_classes = 0
    # cfg.decoder.depth = 3
    # cfg.decoder.eeg_size = cfg.encoder.eeg_size // cfg.decoder.patch_size
    # cfg.codebook.decay_codebook_mov_avg = decay_codebook
    # cfg.codebook.num_tokens = num_codebook_tokens
    # cfg.codebook.emb_dim = codebook_dim
    # cfg.codebook.kmeans_init = quantize_kmeans_init
    # cfg.update(kwargs)
    # cfg.
    # encoder settings
    # encoder_config['EEG_size'] = EEG_size
    # encoder_config['num_classes'] = 0
    # encoder_config['use_mean_pooling'] = False
    # decoder settings
    # decoder_config['EEG_size'] = EEG_size // decoder_config['patch_size']
    # decoder_config['patch_size'] = 1
    # decoder_config['in_chans'] = code_dim
    # decoder_config['num_classes'] = 0
    # decoder_config['depth'] = 3
    # decoder_out_dim = 200

    model = VQNSP(cfg)

    if as_tokenizer:
        # assert weights_path is not None

        _load_weights(model, weights_path)
    return model


# pretrained=False,
#                                         pretrained_weight=None,
#                                         as_tokenizer=False,
#                                         eeg_size=1600,
#                                         decay_codebook: float = 0.99,
#                                         num_codebook_tokens: int = 8192,
#                                         codebook_dim: int = 32,
#                                         **kwargs
@register_model
def vqnsp_encoder_large_decoder_3x200x24(cfg: Optional[ConfigVQNSP] = None,
                                         weights_path: Optional[str] = None,
                                         as_tokenizer: bool = False,
                                         # eeg_size: int = 1600,
                                         # decay_codebook: float = DEFAULT_CODEBOOK_DECAY,
                                         # num_codebook_tokens: int = DEFAULT_CODEBOOK_SIZE,
                                         # codebook_dim: int = DEFAULT_CODEBOOK_EMBED_DIM,
                                         # quantize_kmeans_init: bool =True,
                                         **kwargs) -> VQNSP:
    # encoder_config, decoder_config = get_vqnsp_default_cfg(), get_vqnsp_default_cfg()

    if cfg is None:
        cfg = get_cfg_vqnsp_large_3x200x12(**kwargs)
    # encoder settings
    # cfg = ConfigVQNSP()
    # cfg.encoder.eeg_size = eeg_size
    # cfg.encoder.num_classes = 0
    # cfg.encoder.depth = 24
    # cfg.decoder.num_classes = 0
    # cfg.decoder.depth = 3
    # cfg.decoder.eeg_size = cfg.encoder.eeg_size // cfg.decoder.patch_size
    # cfg.codebook.decay_codebook_mov_avg = decay_codebook
    # cfg.codebook.num_tokens = num_codebook_tokens
    # cfg.codebook.emb_dim = codebook_dim
    # cfg.codebook.kmeans_init = quantize_kmeans_init
    # cfg.update(kwargs)
    # encoder_config['EEG_size'] = EEG_size
    # encoder_config['num_classes'] = 0
    # encoder_config['depth'] = 24
    # # encoder_config['use_mean_pooling'] = False
    # # decoder settings
    # decoder_config['EEG_size'] = EEG_size // decoder_config['patch_size']
    # decoder_config['patch_size'] = 1
    # decoder_config['in_chans'] = code_dim
    # decoder_config['num_classes'] = 0
    # decoder_config['depth'] = 3
    # decoder_out_dim = 200

    model = VQNSP(cfg)

    if as_tokenizer:
        _load_weights(model, weights_path)
    return model


# def get_vqnsp_default_cfg() -> ConfigVQNSP:
#     """A factory Get default parameters for VQNSP model."""
#     cfg = ConfigVQNSP()
#     cfg.encoder.eeg_size = 1600
#     cfg.decoder.eeg_size = 1600
#     cfg.encoder.patch_size = 200
#     cfg.decoder.patch_size = 200
#     cfg.encoder.in_chans = 1
#
#     return dict(EEG_size=1600, patch_size=200, in_chans=1, num_classes=1000,
#                 embed_dim=200, depth=12, num_heads=10,
#                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
#                 attn_drop_rate=0., drop_path_rate=0.,
#                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                 init_values=0., use_abs_pos_emb=True,
#                 use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
#                 use_mean_pooling=True, init_scale=0.001)

def _load_weights(model: VQNSP, weights_path: str):
    if not (weights_path or Path(weights_path).is_file()):
        raise FileNotFoundError(f"Weights file {weights_path} does not exist")
    model.encoder.use_mean_pooling = False  # TODO: must be changed in config
    model.decoder.use_mean_pooling = False  # TODO: must be changed in config
    if weights_path.startswith('https'):
        weights = torch.hub.load_state_dict_from_url(weights_path, map_location='cpu', check_hash=True)
    else:
        weights = torch.load(weights_path, map_location='cpu', weights_only=False)

    if 'model' in weights:
        weights = weights['model']
    else:
        weights = weights["state_dict"]
    keys = list(weights.keys())

    for k in keys:
        if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
            del weights[k]

    def replace_names(key_):
        key_ = key_.replace(".fc_norm.", ".norm.")
        key_ = key_.replace('quantize.', "quantizer.")
        key_ = key_.replace('.embedding.', ".embedder.")
        return key_

    weights = {replace_names(k): v for k, v in weights.items()}

    model.load_state_dict(weights)


if __name__ == '__main__':
    pass
