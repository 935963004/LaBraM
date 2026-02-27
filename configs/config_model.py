import enum
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, List, Union

from configs.config_base import ParamsBase, ConfigBase
from data.eeg_consts import MIN_LEN_SEC, DEFAULT_SAMPLING_RATE


class FeaturesType(str, enum.Enum):
    PATCH_TOKENS = "patch_tokens"
    CLS_TOKEN = "cls_token"
    QUANTIZE_TOKENS = "quantize_tokens"
    BAG_OF_CODES = "codebook_ind"

    @classmethod
    def values(cls):
        return [member.value for member in cls]

    @classmethod
    def names(cls):
        return [member.name for member in cls]


class ClassifierTypes(str, enum.Enum):
    LINEAR = "linear"
    MLP = "mlp"


class NormLayer(str, enum.Enum):
    LAYER_NORM = "LayerNorm"


FEATURES_LIST = List[Union[str, FeaturesType]]
DEFAULT_DEPTH_ENCODER = 12
DEFAULT_CODEBOOK_SIZE = 8192
DEFAULT_CODEBOOK_EMBED_DIM = 32
DEFAULT_CODEBOOK_DECAY = 0.99


@dataclass
class ConfigCodebookQuantizer(ConfigBase):
    num_tokens: int = DEFAULT_CODEBOOK_SIZE  # Number of tokens(words) in the codebook
    emb_dim: int = DEFAULT_CODEBOOK_EMBED_DIM  # Dimension of the tokens(words) in the codebook
    betta_quantize_err: float = 1.0  # weight of the quantization loss/error
    decay_codebook_mov_avg: float = DEFAULT_CODEBOOK_DECAY  # decay rate of the moving average of training update the codebook
    statistic_code_usage: bool = True  # Reset the codebook usage statistic info before each epoch/eval
    eps_smoothing: float = 1e-5  # Epsilon smoothing for codebook token update
    kmeans_init: bool = False  # Flag to enable k-means initialization of the codebook with zeros ELSE random initialization
    codebook_init_path: str = ''  # Path to load the codebook weights from the file
    update_codebook: bool = True  # Flag to enable updating the codebook statistics over training epoch or eval


@dataclass
class ConfigNeuralTransformer(ParamsBase):
    eeg_size: int = MIN_LEN_SEC  # 1600
    patch_size: int = DEFAULT_SAMPLING_RATE
    in_chans: int = 1
    out_chans: int = 8
    num_classes: int = 1000
    embed_dim: int = 200
    depth: int = DEFAULT_DEPTH_ENCODER
    num_heads: int = 10
    mlp_ratio: float = 4.
    qkv_bias: bool = False
    qk_norm: Optional[str] = None  # "LayerNorm"
    qk_scale: Optional[float] = None
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.
    norm_layer: str = "LayerNorm"
    norm_eps: float = 1e-6
    init_values: float = 0.
    use_abs_pos_emb: bool = True
    use_rel_pos_bias: bool = False
    use_shared_rel_pos_bias: bool = False
    use_mean_pooling: bool = True
    init_scale: float = 0.001
    classifier_type: Optional[str] = ClassifierTypes.LINEAR.value

    def check_valid(self):
        pass
        # if isinstance(self.classifier_type, str):
        #     self.classifier_type = ClassifierTypes(self.classifier_type)


def get_vqnsp_neural_transformer_default_cfg(is_encoder: bool) -> ConfigNeuralTransformer:
    in_chans = 1 if is_encoder else DEFAULT_CODEBOOK_EMBED_DIM
    depth = DEFAULT_DEPTH_ENCODER if is_encoder else 3
    patch_size = DEFAULT_SAMPLING_RATE if is_encoder else 1
    cfg = ConfigNeuralTransformer(eeg_size=1600,
                                  patch_size=patch_size,
                                  in_chans=in_chans,
                                  num_classes=0,
                                  embed_dim=200,
                                  depth=depth,
                                  num_heads=10,
                                  mlp_ratio=4.,
                                  qkv_bias=True,
                                  qk_scale=None,
                                  drop_rate=0.,
                                  attn_drop_rate=0.,
                                  drop_path_rate=0.,
                                  norm_layer="LayerNorm",
                                  norm_eps=1e-6,
                                  init_values=0.,
                                  use_abs_pos_emb=True,
                                  use_rel_pos_bias=False,
                                  use_shared_rel_pos_bias=False,
                                  use_mean_pooling=True,
                                  init_scale=0.001,
                                  )
    return cfg


# DEFAULT_VQNSP_DECODER_CONFIG = DEFAULT_VQNSP_ENCODER_CONFIG.update(in_chans=DEFAULT_CODEBOOK_EMBED_DIM)

def get_codebook_default_cfg() -> ConfigCodebookQuantizer:
    cfg = ConfigCodebookQuantizer(num_tokens=DEFAULT_CODEBOOK_SIZE,
                                  emb_dim=DEFAULT_CODEBOOK_EMBED_DIM,
                                  decay_codebook_mov_avg=0.99,
                                  kmeans_init=True)
    return cfg

@dataclass
class ConfigVQNSP(ParamsBase):
    encoder: ConfigNeuralTransformer = field(
        default_factory=partial(get_vqnsp_neural_transformer_default_cfg, is_encoder=True)
    )
    decoder: ConfigNeuralTransformer = field(
        default_factory=partial(get_vqnsp_neural_transformer_default_cfg, is_encoder=False)
    )  # name: str = 'vqnsp_encoder_base_decoder_3x200x12'
    # n_embed: int = 8192
    # embed_dim: int = 32
    # decay: float = 0.99
    # quantize_kmeans_init: bool =True
    codebook: ConfigCodebookQuantizer = field(default_factory=get_codebook_default_cfg)
    out_dim: int = DEFAULT_SAMPLING_RATE

    def __init__(self,
                 encoder: Optional[ConfigNeuralTransformer] = None,
                 decoder: Optional[ConfigNeuralTransformer] = None,
                 codebook: Optional[ConfigCodebookQuantizer] = None,
                 out_dim: Optional[int] = None,
                 in_chans: Optional[int] = None,
                 embed_dim: Optional[int] = None,
                 num_tokens: Optional[int] = None):
        super().__init__()
        self.encoder = get_vqnsp_neural_transformer_default_cfg(is_encoder=True) if encoder is None else encoder
        self.decoder = get_vqnsp_neural_transformer_default_cfg(is_encoder=False) if decoder is None else decoder
        self.codebook = get_codebook_default_cfg() if codebook is None else codebook

        if in_chans is not None:
            self.encoder.in_chans = in_chans
        elif encoder is None:
            self.encoder.in_chans = 1

        if embed_dim is not None:
            self.decoder.in_chans = embed_dim
        elif decoder is None:
            self.decoder.in_chans = DEFAULT_CODEBOOK_EMBED_DIM

        if num_tokens is not None:
            self.codebook.num_tokens = num_tokens
        elif codebook is None:
            self.codebook.num_tokens = DEFAULT_CODEBOOK_SIZE

        if embed_dim is not None:
            self.codebook.emb_dim = embed_dim
        elif codebook is None:
            self.codebook.emb_dim = DEFAULT_CODEBOOK_EMBED_DIM

        self.out_dim = out_dim if out_dim is not None else DEFAULT_SAMPLING_RATE

    def check_valid(self):
        pass

@dataclass
class ConfigEEGClassifier(ParamsBase):
    num_classes: int = 1
    classifier_type: str = ClassifierTypes.LINEAR.value
    drop_rate: float = 0.0
    feature_space: List[str] = field(default_factory=lambda: [FeaturesType.PATCH_TOKENS.value])
    features_emb_dim: int = 128
    norm_embedding: bool = True
    update_codebook: bool = True
    linear_embedding: bool = False

    name_encoder: str = 'labram_base_patch200_200'
    name_vqnsp: str = 'vqnsp_encoder_base_decoder_3x200x12'
    weights_encoder_path: Optional[str] = None
    weights_vqnsp_path: Optional[str] = None

    vqnsp: ConfigVQNSP = field(default_factory=lambda: ConfigVQNSP())
    encoder: ConfigNeuralTransformer = field(default_factory=lambda: ConfigNeuralTransformer())

    def __post_init__(self):
        # if isinstance(self.classifier_type, str):
        #     self.classifier_type = ClassifierTypes(self.classifier_type)
        #
        # self.feature_space = [FeaturesType(feature) if isinstance(feature, str) else feature
        #                       for feature in self.feature_space]


        super().__post_init__()


    def check_valid(self) -> None:
       pass