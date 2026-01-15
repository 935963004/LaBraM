"""
This module defines the VQNSP class, a neural network model aimed at enabling effective
learning and quantization of EEG data representations. The model comprises an encoder,
a vector quantization mechanism, and decoding tasks for prediction.

The VQNSP class incorporates configurable encoder and decoder components, customizable
quantization parameters, and methods for processing and reconstructing EEG input data
in the magnitude and phase domains.

Classes:
- VQNSP: Defines the model architecture, including encoding, quantization, and decoding
  logic.

Attributes:
- encoder_config: Configuration dictionary for the encoder module.
- decoder_config: Configuration dictionary for the decoder module.
- n_embed: Number of embedding vectors in the codebook.
- embed_dim: Dimension of each embedding vector in the quantization space.
- decay: Decay rate for the exponential moving average in quantization.
- quantize_kmeans_init: Flag to enable k-means initialization in the quantizer.
- decoder_out_dim: Output dimension for decoding tasks.

"""
from typing import Dict, List, Tuple
import torch
from einops import rearrange
from timm.layers import trunc_normal_
from torch import nn, Tensor

from models.codebook_embedding import NormEMAVectorQuantizer
from models.neural_transformer import NeuralTransformer


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

        # quantizer that matches encoder patches to codebook words
        self.quantizer = NormEMAVectorQuantizer(
            n_embed=n_embed,
            embedding_dim=embed_dim,
            beta=1.0,
            kmeans_init=quantize_kmeans_init,
            decay=decay,
        )

        self.patch_size = encoder_config['patch_size']
        self.token_shape = (62, encoder_config['EEG_size'] // self.patch_size)

        self.decoder_out_dim = decoder_out_dim

        # embedding into quantization space
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim) # for quantize
        )

        # MLP prediction heads:
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        ) # magnitude prediction

        self.decode_task_layer_angle = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        ) # phase prediction

        self.kwargs = kwargs

        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)
        self.decode_task_layer_angle.apply(self._init_weights)

        # self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

    def forward(self, x: Tensor, input_chans: List[str] = None, **kwargs) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        x: shape [B, N, T]
        """

        x = rearrange(x, 'B N (A T) -> B N A T', T=200)

        encoder_out = self.encode(x, input_chans)

        # predict amplitude and phase from quantized tokens
        recon_amplitude, recon_phase = self.decode(encoder_out['quantize_tokens'], input_chans)
        recon_out = {'recon_amplitude': recon_amplitude,
                     'recon_phase': recon_phase}

        return recon_out, encoder_out

    def encode(self, x: Tensor, input_chans: List[str] = None):
        """
        Encodes the input EEG data and processes it through an encoder and quantization module.

        The method takes input EEG signal in the form of a tensor and processes it using an encoder,
        producing intermediate and final output features. It applies quantization to the
        encoder features and returns the encoded output along with quantized representations
        and associated loss.

        Args:
            x: Tensor of shape (batch_size, n_channels, n_samples, times), representing
                the input EEG data to be encoded.
            input_chans: Optional parameter specifying the number of input channels. If not
                provided, this will be inferred from the input shape.

        Returns:
            Dictionary containing:
                - pred_class: Predicted class probabilities as derived by the encoder.
                - patch_tokens: Tensor representing the tokens extracted from
                  input EEG patches by the encoder.
                - cls_token:  representing the class token = time window of N patches
                - quantize_tokens: Tensor containing quantized representations of
                  encoded features.
                - codebook_ind: Indices representing the selected codebook entries
                  for quantization of each patch.
                - quantize_loss: Loss value associated with the quantization process.
        """
        batch_size, n_channels, n_samples, times = x.shape
        encoder_out = self.encoder(x, input_chans)
        encoder_features = encoder_out['patch_tokens']  # b, num_patches, embed_dim
        codebook_ind, quantize_loss, quantize_tokens = self.quantize_enc_features(encoder_features,
                                                                                  n_channels)
        encoder_out['quantize_tokens'] = quantize_tokens
        encoder_out['codebook_ind'] = codebook_ind
        encoder_out['quantize_loss'] = quantize_loss
        return encoder_out

    def quantize_enc_features(self, encoder_features: Tensor, n_channels: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Quantizes the features extracted from the encoder if codebook tokens.
        Parameters:
        encoder_features: Tensor
            Input features generated by the encoder.
        n_channels: int
            Number of channels used for grouping input features during quantization.

        Returns:
            - codebook_ind: The indices of the quantized latent embeddings in the codebook.
            - quantize_error: The reconstruction L2 error introduced during quantization.
            - quantize_tokens: The quantized representation of the input features.
        """
        # embedding into quantization space
        with torch.amp.autocast(self.device):
            quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        n_patches = quantizer_features.shape[1] // n_channels  # n_patches

        # reshape for quantizer: (batch, codebook_dim, n_channels, n_patches)
        quantizer_features = rearrange(quantizer_features,
                                       'b (h w) c -> b c h w',
                                       h=n_channels,
                                       w=n_patches)

        # apply quantization
        quantize_tokens, quantize_error, codebook_ind = self.quantizer(quantizer_features)
        return codebook_ind, quantize_error, quantize_tokens

    def decode(self, quantize_patches: Tensor, input_chans: List[str] = None, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Decode quantized patches to reconstruct amplitude and phase representations.
        """
        decoder_out = self.decoder(quantize_patches, input_chans)
        decoder_features = decoder_out['patch_tokens']
        recon_amplitude = self.decode_task_layer(decoder_features)
        recon_angle = self.decode_task_layer_angle(decoder_features)
        return recon_amplitude, recon_angle

    @staticmethod
    def _init_weights(m):
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



