import math
from typing import Optional, List, Dict, Set

import torch
from timm.layers import trunc_normal_
from torch import nn as nn, Tensor

from configs import ConfigNeuralTransformer, ClassifierTypes
from layers.attention_blocks import Block
from layers.mlp_blocks import MlpClassifier
from layers.norm_layers import get_norm_layer
from layers.patch_conv_blocks import TemporalConv, PatchEmbed


class NeuralTransformer(nn.Module):
    def __init__(self, cfg: ConfigNeuralTransformer, **kwargs):
        """Defines transformer with temporal convolution or patch embedding"""
        super().__init__()
        self.cfg = cfg.update(**kwargs)
        # self.num_classes = self.cfg.num_classes
        # self.embed_dim = self.cfg.embed_dim
        # self.num_features = self.cfg.embed_dim  # num_features for consistency with other models
        # self.use_mean_pooling = use_mean_pooling
        # To identify whether it is neural tokenizer or neural decoder.
        # For the neural decoder, use linear projection (PatchEmbed) to project codebook dimension to hidden dimension.
        # Otherwise, use TemporalConv to extract temporal features from EEG signals.
        # qk_norm = get_norm_layer(self.cfg.qk_norm, dim=self.cfg.embed_dim, eps=self.cfg.norm_eps)
        # norm_layer = get_norm_layer(self.cfg.norm_layer, dim=self.cfg.embed_dim, eps=self.cfg.norm_eps)
        if self.cfg.in_chans == 1:
            self.patch_embed = TemporalConv(out_chans=self.cfg.out_chans)
        else:
            self.patch_embed = PatchEmbed(EEG_size=self.cfg.eeg_size,
                                          patch_size=self.cfg.patch_size,
                                          in_chans=self.cfg.in_chans,
                                          embed_dim=self.cfg.embed_dim)

        self.patch_size = self.cfg.patch_size

        self.time_window = self.cfg.eeg_size // self.patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.cfg.use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, self.cfg.embed_dim),
                                          requires_grad=True)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, self.cfg.embed_dim),
                                       requires_grad=True)
        self.pos_drop = nn.Dropout(p=self.cfg.drop_rate)
        self.rel_pos_bias = None
        dpr = [x.item() for x in
               torch.linspace(0, self.cfg.drop_path_rate, self.cfg.depth)]  # stochastic depth decay rule
        # self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=self.cfg.embed_dim,
                num_heads=self.cfg.num_heads,
                mlp_ratio=self.cfg.mlp_ratio,
                qkv_bias=self.cfg.qkv_bias,
                qk_norm=self.cfg.qk_norm,
                qk_scale=self.cfg.qk_scale,
                drop=self.cfg.drop_rate,
                attn_drop=self.cfg.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=self.cfg.norm_layer,
                init_values=self.cfg.init_values,
                norm_eps=self.cfg.norm_eps,
                window_size=None)
            for i in range(self.cfg.depth)])

        self.norm = get_norm_layer(self.cfg.norm_layer,
                                   self.cfg.embed_dim)  # nn.Identity() if use_mean_pooling else norm_layer(embed_dim)

        if self.cfg.num_classes == 0:
            self.head = nn.Identity()
        elif self.cfg.classifier_type == ClassifierTypes.LINEAR:
            self.head = nn.Linear(self.cfg.embed_dim, self.cfg.num_classes)
        elif self.cfg.classifier_type == ClassifierTypes.MLP:
            self.head = MlpClassifier(self.cfg.embed_dim,
                                      act_layer=nn.GELU,
                                      num_classes=self.cfg.num_classes,
                                      drop=self.cfg.drop_rate)
        else:
            raise ValueError(f'Invalid classifier type: {self.cfg.classifier_type}')

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(self.cfg.init_scale)
            self.head.bias.data.mul_(self.cfg.init_scale)

    def forward(self,
                x: Tensor,
                input_chans: List[str] = None,
                **kwargs) -> Dict[str, Tensor]:
        """
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        """
        x = self.forward_features(x,
                                  input_chans=input_chans,
                                  # return_patch_tokens=return_patch_tokens,
                                  # return_all_tokens=return_all_tokens,
                                  **kwargs)
        patch_tokens = x[:, 1:]
        cls_token = x[:, 0]

        if self.cfg.use_mean_pooling:
            features_pred = patch_tokens.mean(1)
        else:
            features_pred = cls_token
        pred_class = self.head(features_pred)

        output = {'pred_class': pred_class,
                  'patch_tokens': patch_tokens,
                  'cls_token': cls_token}
        return output

    def forward_features(self,
                         x: Tensor,
                         input_chans: Optional[List[str]] = None,
                         is_last_global_att: bool = False, **kwargs) -> Tensor:
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1,
                                                                                                                    2)
            pos_embed = torch.cat((pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(
                1, 2)
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)

        for blk in self.blocks[:-1]:
            x = blk(x, rel_pos_bias=None)

        x = self.blocks[-1](x, rel_pos_bias=None, global_att_only=is_last_global_att)

        return self.norm(x)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {'pos_embed', 'cls_token', 'time_embed'}

    def reset_classifier(self, num_classes: int, global_pool=''):
        self.cfg.num_classes = num_classes
        self.head = nn.Linear(self.cfg.embed_dim, num_classes) if num_classes > 0 else nn.Identity()



    def get_intermediate_layers(self, x, use_last_norm=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)

        return features
