from typing import Dict, List, Tuple, Any, Optional, Set
import torch
import enum
import warnings
from einops import rearrange
from timm.layers import trunc_normal_
from torch import nn, Tensor

from models.codebook_quantizer import NormEMAVectorQuantizer
from models.vqnsp_model import VQNSP, NeuralTransformer
from layers.mlp_blocks import MlpClassifier
from layers.embedding_blocks import FeatureEmbedder, CodeBookBagEmbedder


class FeaturesTypes(enum.Enum):
    PATCH_TOKENS = "patch_tokens"
    CLS_TOKEN = "cls_token"
    QUANTIZE_TOKENS = "quantize_tokens"
    BAG_OF_CODES = "codebook_ind"

    @classmethod
    def values(cls):
        return [member.value for member in cls]

    @classmethod
    def names(cls):
        return [member.name  for member in cls]

class NeurolCodebookClassifier(nn.Module):
    def __init__(self,
                 vqnsp_model: VQNSP,
                 encoder_model: NeuralTransformer,
                 num_classes: int = 10,
                 classifier_type: str = 'linear', # or 'mlp'
                 drop_rate: float = 0.0,
                 feature_space: List[FeaturesTypes] = [FeaturesTypes.PATCH_TOKENS.value],
                 features_emb_dim: int = 128,
                 norm_embedding: bool = True,
                 update_codebook: bool = True,
                 **kwargs
                 ):
        super().__init__()
        self.vqnsp_model: VQNSP = vqnsp_model
        self.vqnsp_model.encoder = encoder_model
        self.vqnsp_model.quantizer.update_codebook = update_codebook

        self.feature_space = feature_space
        self.features_emb_dim=features_emb_dim
        self.embedders: nn.ModuleDict = self.build_features_embedders(features_emb_dim)
        self.features_dim = features_emb_dim * len(self.feature_space)
        if self.features_dim == 0:
            raise ValueError("No feature space specified for the classifier model.")

        self.norm_emb = nn.LayerNorm(self.features_dim) if norm_embedding else nn.Identity()
        if num_classes == 0:
            self.classifier_head = nn.Identity()
            warnings.warn("No classifier head specified for the classifier model.")
        elif classifier_type == 'linear':
            self.classifier_head = nn.Linear(self.features_dim, num_classes)
        elif classifier_type == 'mlp':
            self.classifier_head = MlpClassifier(self.features_dim,
                                      act_layer=nn.GELU,
                                      num_classes=num_classes,
                                      drop=drop_rate)
        else:
            raise ValueError(f'Invalid classifier type: {classifier_type}')


    def forward(self, x: Tensor, input_chans: List[str] = None, **kwargs) -> Tensor:
        decoder_out, encoder_out = self.vqnsp_model(x, input_chans, **kwargs)
        features = self.calculate_classify_features(encoder_out, decoder_out)
        features = self.norm_emb(features)
        pred_out = self.classifier_head(features)
        return pred_out, decoder_out, encoder_out

    def build_features_embedders(self, embed_dim: int=128) -> nn.ModuleDict:
        # Build embedders for each feature type
        embedders = nn.ModuleDict()
        if len(self.feature_space) == 0:
            raise ValueError("No feature space specified for the classifier model.")

        for feature_type in self.feature_space:
            if feature_type == FeaturesTypes.PATCH_TOKENS.value:
                embedders[feature_type] = FeatureEmbedder(in_dim=self.vqnsp_model.encoder.embed_dim,
                                                          out_dim=embed_dim,
                                                          reduce_dim=1)
            elif feature_type == FeaturesTypes.QUANTIZE_TOKENS.value:
                embedders[feature_type] = FeatureEmbedder(in_dim=self.vqnsp_model.quantizer.codebook_dim,
                                                          out_dim=embed_dim,
                                                          reduce_dim=1)
            elif feature_type == FeaturesTypes.CLS_TOKEN.value:
                embedders[feature_type] = FeatureEmbedder(self.vqnsp_model.encoder.embed_dim,
                                                          out_dim=embed_dim,
                                                          reduce_dim=None)
            elif feature_type == FeaturesTypes.BAG_OF_CODES.value:
                embedders[feature_type] = CodeBookBagEmbedder(self.vqnsp_model.quantizer.num_tokens,
                                                              out_dim=embed_dim)
            else:
                raise ValueError(f'Invalid feature type: {feature_type} '
                                 f'supported types: {FeaturesTypes.values()}')
        return embedders

    def calculate_classify_features(self,
                                    encoder_features: Dict[str,Tensor],
                                    decoder_features: Dict[str,Tensor] = None) -> Tensor:
        # Concatenate encoder and decoder features along the feature for classification head
        combined_features = []
        for feature_type in self.feature_space:
            if feature_type not in encoder_features.keys():
                raise ValueError(f'Feature type {feature_type} not found in encoder features. '
                                 f'encoder_features includes only:{encoder_features.keys()}')
            elif feature_type not in FeaturesTypes.values():
                raise ValueError(f'Feature type {feature_type} not found in FeaturesTypes enum. '
                                 f'Supported types: {FeaturesTypes.values()}')

            features_emb = self.embedders[feature_type](encoder_features[feature_type])
            if features_emb.shape[-1] != self.features_emb_dim:
                raise ValueError(f'Embeddings of feature type {feature_type} have incorrect shape. {features_emb.shape}')
            combined_features.append(features_emb)

        combined_features = torch.cat(combined_features, dim=-1)
        return combined_features

    def train(self, mode: bool = True, wo_codebook: bool = True):
        super().train(mode)
        if wo_codebook:
            self.vqnsp_model.quantizer.eval()

    @property
    def encoder(self)->NeuralTransformer:
        return self.vqnsp_model.encoder

    @property
    def quantizer(self)->NormEMAVectorQuantizer:
        return self.vqnsp_model.quantizer

    @property
    def decoder(self)->NeuralTransformer:
        return self.vqnsp_model.decoder

    @property
    def patch_size(self)->int:
        return self.encoder.patch_size

    def no_weight_decay(self) -> Set[str]:
        return self.vqnsp_model.no_weight_decay()