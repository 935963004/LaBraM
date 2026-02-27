import warnings
from typing import Dict, List, Set

import torch
from timm.layers import trunc_normal_
from torch import nn, Tensor

from configs import FeaturesType, ConfigEEGClassifier, ClassifierTypes
from layers.embedding_blocks import FeatureEmbedder, CodeBookBagEmbedder
from layers.mlp_blocks import MlpClassifier
from models.codebook_quantizer import NormEMAVectorQuantizer
from models.vqnsp_model import VQNSP, NeuralTransformer




class NeurolCodebookClassifier(nn.Module):
    def __init__(self,
                 vqnsp: VQNSP,
                 encoder: NeuralTransformer,
                 cfg: ConfigEEGClassifier,
                 # num_classes: int = 10,
                 # classifier_type: str = 'linear', # or 'mlp'
                 # drop_rate: float = 0.0,
                 # feature_space: List[FeaturesType] = [FeaturesType.PATCH_TOKENS.value],
                 # features_emb_dim: int = 128,
                 # norm_embedding: bool = True,
                 # update_codebook: bool = True,
                 **kwargs
                 ):
        super().__init__()
        self.vqnsp: VQNSP = vqnsp
        self.vqnsp.encoder = encoder
        self.vqnsp.quantizer.update_codebook = cfg.update_codebook

        self.feature_space = cfg.feature_space
        self.features_emb_dim = cfg.features_emb_dim
        self.embedders: nn.ModuleDict = self.build_features_embedders(self.features_emb_dim,
                                                                      is_linear=cfg.linear_embedding)
        self.features_dim = self.features_emb_dim * len(self.feature_space)
        if self.features_dim == 0:
            raise ValueError("No feature space specified for the classifier model.")

        self.norm_emb = nn.LayerNorm(self.features_dim) if cfg.norm_embedding else nn.Identity()
        if cfg.num_classes == 0:
            self.classifier_head = nn.Identity()
            warnings.warn("No classifier head specified for the classifier model.")
        elif cfg.classifier_type == ClassifierTypes.LINEAR:
            self.classifier_head = nn.Linear(self.features_dim, cfg.num_classes)
        elif cfg.classifier_type == ClassifierTypes.MLP:
            self.classifier_head = MlpClassifier(self.features_dim,
                                      act_layer=nn.GELU,
                                                 num_classes=cfg.num_classes,
                                                 drop=cfg.drop_rate)
        else:
            raise ValueError(f'Invalid classifier type: {cfg.classifier_type}')

    def forward(self, x: Tensor, input_chans: List[str] = None, **kwargs) -> [Tensor, Tensor, Tensor]:
        decoder_out, encoder_out = self.vqnsp(x, input_chans, **kwargs)
        features = self.calculate_classify_features(encoder_out, decoder_out)
        features = self.norm_emb(features)
        pred_out = self.classifier_head(features)
        return pred_out, decoder_out, encoder_out

    def build_features_embedders(self, embed_dim: int=128, is_linear: bool = False) -> nn.ModuleDict:
        # Build embedders for each feature type
        embedders = nn.ModuleDict()
        if len(self.feature_space) == 0:
            raise ValueError("No feature space specified for the classifier model.")

        for feature_type in self.feature_space:
            if feature_type == FeaturesType.PATCH_TOKENS.value:
                embedders[feature_type] = FeatureEmbedder(in_dim=self.vqnsp.encoder.cfg.embed_dim,
                                                          out_dim=embed_dim,
                                                          is_linear=is_linear,
                                                          reduce_dim=1)
            elif feature_type == FeaturesType.QUANTIZE_TOKENS.value:
                embedders[feature_type] = FeatureEmbedder(in_dim=self.vqnsp.quantizer.cfg.codebook_dim,
                                                          out_dim=embed_dim,
                                                          is_linear=is_linear,
                                                          reduce_dim=1)
            elif feature_type == FeaturesType.CLS_TOKEN.value:
                embedders[feature_type] = FeatureEmbedder(self.vqnsp.encoder.cfg.embed_dim,
                                                          out_dim=embed_dim,
                                                          is_linear=is_linear,
                                                          reduce_dim=None)
            elif feature_type == FeaturesType.BAG_OF_CODES.value:
                embedders[feature_type] = CodeBookBagEmbedder(self.vqnsp.quantizer.cfg.num_tokens,
                                                              out_dim=embed_dim)
            else:
                raise ValueError(f'Invalid feature type: {feature_type} '
                                 f'supported types: {FeaturesType.values()}')
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
            elif feature_type not in FeaturesType.values():
                raise ValueError(f'Feature type {feature_type} not found in FeaturesTypes enum. '
                                 f'Supported types: {FeaturesType.values()}')

            features_emb = self.embedders[feature_type](encoder_features[feature_type])
            if features_emb.shape[-1] != self.features_emb_dim:
                raise ValueError(f'Embeddings of feature type {feature_type} have incorrect shape. {features_emb.shape}')
            combined_features.append(features_emb)

        combined_features = torch.cat(combined_features, dim=-1)
        return combined_features

    def train(self, mode: bool = True, wo_codebook: bool = True):
        super().train(mode)
        if wo_codebook:
            self.vqnsp.quantizer.eval()
            self.vqnsp.quantizer.update_codebook = False
            self.vqnsp.decoder.eval()
            self.vqnsp.quantizer.eval()
            self.vqnsp.encode_task_layer.eval()
            self.vqnsp.decode_task_layer.eval()
            self.vqnsp.decode_task_layer_angle.eval()


    @property
    def encoder(self)->NeuralTransformer:
        return self.vqnsp.encoder

    @property
    def quantizer(self)->NormEMAVectorQuantizer:
        return self.vqnsp.quantizer

    @property
    def decoder(self)->NeuralTransformer:
        return self.vqnsp.decoder

    @property
    def patch_size(self)->int:
        return self.encoder.patch_size

    def no_weight_decay(self) -> Set[str]:
        return self.vqnsp.no_weight_decay()
