# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model

from modeling_finetune import NeuralTransformer
from norm_ema_quantizer import NormEMAVectorQuantizer

class VQNSP(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_embed=8192, 
                 embed_dim=32,
                 decay=0.99,
                 quantize_kmeans_init=True,
                 decoder_out_dim=200,
                 smooth_l1_loss = False,
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        if decoder_config['in_chans'] != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config['in_chans']} to {embed_dim}")
            decoder_config['in_chans'] = embed_dim

        # encoder & decode params
        print('Final encoder config', encoder_config)
        self.encoder = NeuralTransformer(**encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder = NeuralTransformer(**decoder_config)
                
        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )
        
        self.patch_size = encoder_config['patch_size']
        self.token_shape = (62, encoder_config['EEG_size'] // self.patch_size)

        self.decoder_out_dim = decoder_out_dim

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim) # for quantize
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.decode_task_layer_angle = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )

        self.kwargs = kwargs
        
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)
        self.decode_task_layer_angle.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed', 
                'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed'}

    @property
    def device(self):
        return self.decoder.cls_token.device
        
    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, input_chans=None, **kwargs):
        quantize, embed_ind, loss = self.encode(data, input_chans=input_chans)
        output = {}
        output['token'] = embed_ind.view(data.shape[0], -1)
        output['input_img'] = data
        output['quantize'] = rearrange(quantize, 'b d a c -> b (a c) d')

        return output

    def encode(self, x, input_chans=None):
        batch_size, n, a, t = x.shape
        encoder_features = self.encoder(x, input_chans, return_patch_tokens=True)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        N = to_quantizer_features.shape[1]
        h, w = n, N // n

        to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, loss
    
    def decode(self, quantize, input_chans=None, **kwargs):
        # reshape tokens to feature maps for patch embed in decoder
        # quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=self.token_shape[0], w=self.token_shape[1])
        decoder_features = self.decoder(quantize, input_chans, return_patch_tokens=True)
        rec = self.decode_task_layer(decoder_features)
        rec_angle = self.decode_task_layer_angle(decoder_features)
        return rec, rec_angle
    
    def get_codebook_indices(self, x, input_chans=None, **kwargs):
        # for LaBraM pre-training
        return self.get_tokens(x, input_chans, **kwargs)['token']
    
    def calculate_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c')
        rec_loss = self.loss_fn(rec, target)
        return rec_loss
    
    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / std
        return x

    def forward(self, x, input_chans=None, **kwargs):
        """
        x: shape [B, N, T]
        """

        x = rearrange(x, 'B N (A T) -> B N A T', T=200)
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        amplitude = self.std_norm(amplitude)
        angle = torch.angle(x_fft)
        angle = self.std_norm(angle)

        quantize, embed_ind, emb_loss = self.encode(x, input_chans)
        
        xrec, xrec_angle = self.decode(quantize, input_chans)
        rec_loss = self.calculate_rec_loss(xrec, amplitude)
        rec_angle_loss = self.calculate_rec_loss(xrec_angle, angle)
        loss = emb_loss + rec_loss + rec_angle_loss

        log = {}
        split="train" if self.training else "val"
        log[f'{split}/quant_loss'] = emb_loss.detach().mean()
        log[f'{split}/rec_loss'] = rec_loss.detach().mean()
        log[f'{split}/rec_angle_loss'] = rec_angle_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, log

def get_model_default_params():
    return dict(EEG_size=1600, patch_size=200, in_chans=1, num_classes=1000, embed_dim=200, depth=12, num_heads=10,  
                             mlp_ratio=4., qkv_bias=True,  qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                             norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., use_abs_pos_emb=True, 
                             use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)

@register_model
def vqnsp_encoder_base_decoder_3x200x12(pretrained=False, pretrained_weight=None, as_tokenzer=False, EEG_size=1600, 
                                            n_code=8192, code_dim=32, **kwargs):
    encoder_config, decoder_config = get_model_default_params(), get_model_default_params()

    # encoder settings
    encoder_config['EEG_size'] = EEG_size
    encoder_config['num_classes'] = 0
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
        model.load_state_dict(weights)
    return model

@register_model
def vqnsp_encoder_large_decoder_3x200x24(pretrained=False, pretrained_weight=None, as_tokenzer=False, EEG_size=1600, 
                                            n_code=8192, code_dim=32, **kwargs):
    encoder_config, decoder_config = get_model_default_params(), get_model_default_params()

    # encoder settings
    encoder_config['EEG_size'] = EEG_size
    encoder_config['num_classes'] = 0
    encoder_config['depth'] = 24
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
        model.load_state_dict(weights)
    return model


if __name__ == '__main__':
    pass






