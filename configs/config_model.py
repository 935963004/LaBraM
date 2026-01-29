from dataclasses import dataclass, field
from typing import Optional, List
import enum
from configs.config_base import ParamsBase, ConfigBase

class FeaturesType(str, enum.Enum):
    PATCH_TOKENS = "patch_tokens"
    CLS_TOKEN = "cls_token"
    QUANTIZE_TOKENS = "quantize_tokens"
    BAG_OF_CODES = "codebook_ind"

class ClassifierTypes(str, enum.Enum):
    LINEAR = "linear"
    MLP = "mlp"

FEATURES_LIST = List[FeaturesType]

@dataclass
class ConfigNeuralTransformer(ParamsBase):
    eeg_size: int = 1600
    patch_size: int = 200
    in_chans: int = 1
    out_chans: int = 8
    num_classes: int = 1000
    embed_dim: int = 200
    depth: int = 12
    num_heads: int = 10
    mlp_ratio: float = 4.
    qkv_bias: bool = False
    qk_norm: Optional[str] = None
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
    classifier_type: Optional[ClassifierTypes] = ClassifierTypes.LINEAR

    def check_valid(self):
        if isinstance(self.classifier_type, str):
            self.classifier_type = ClassifierTypes(self.classifier_type)

DEFAULT_VQNSP_ENCODER_CONFIG = ConfigNeuralTransformer(eeg_size=1600,
                                                       patch_size=200,
                                                       in_chans=1,
                                                       num_classes=1000,
                                                       embed_dim=200,
                                                       depth=12,
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
                                                       init_scale=0.001)


@dataclass
class ConfigVQNSPMode(ParamsBase):
    encoder_config : ConfigNeuralTransformer = field(default_factory=DEFAULT_VQNSP_ENCODER_CONFIG)
    decoder_config : ConfigNeuralTransformer = field(default_factory=DEFAULT_VQNSP_ENCODER_CONFIG)
    n_embed: int = 8192
    embed_dim: int = 32
    decay: float = 0.99
    quantize_kmeans_init: bool =True
    decoder_out_dim: int = 200

    def check_valid(self):
        pass

@dataclass
class ConfigEEGClassifierModel(ParamsBase):
    num_classes: int = 10
    classifier_type: ClassifierTypes = ClassifierTypes.LINEAR
    drop_rate: float = 0.0
    feature_space: FEATURES_LIST = field(default_factory=[FeaturesType.PATCH_TOKENS])
    features_emb_dim: int = 128
    norm_embedding: bool = True
    update_codebook: bool = True


    def __post_init__(self):
        if isinstance(self.classifier_type, str):
            self.classifier_type = ClassifierTypes(self.classifier_type)

        self.feature_space = [FeaturesType(feature) if isinstance(feature, str) else feature
                              for feature in self.feature_space]


        super().__post_init__()


    def check_valid(self) -> None:
       pass