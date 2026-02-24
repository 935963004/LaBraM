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
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from configs import ConfigCodebookQuantizer


class NormEMAVectorQuantizer(nn.Module):
    # def __init__(self,
    #              n_embed: int=8192,
    #              embedding_dim: int=32,
    #              beta: float = 1.0,
    #              decay: float = 0.99,
    #              eps: float = 1e-5,
    #              statistic_code_usage: bool = True,
    #              kmeans_init: bool = False,
    #              codebook_init_path: str = '',
    #              update_codebook: bool = True):
    def __init__(self, cfg: ConfigCodebookQuantizer):
        super().__init__()
        self.cfg = cfg
        # self.codebook_dim = embedding_dim
        # self.num_tokens = self.cfg.num_tokens
        # self.beta = beta
        # self.decay = self.cfg.decay_codebook_mov_avg

        # learnable = True if orthogonal_reg_weight > 0 else False
        self.embedder = EmbeddingEMA(self.cfg.num_tokens,
                                     self.cfg.emb_dim,
                                     self.cfg.decay_codebook_mov_avg,
                                     self.cfg.eps_smoothing,
                                     self.cfg.kmeans_init,
                                     self.cfg.codebook_init_path)
        self.embedder.update = self.cfg.update_codebook
        # self.statistic_code_usage = self.cfg.statistic_code_usage
        if self.cfg.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.cfg.num_tokens))
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    def forward(self, z: Tensor) -> Tensor:
        """
        Forward pass for the codebook embedding module.
        Args:
            z (Tensor): Input patches of shape (batch, codebook_dim, n_channels, n_patches).

        Returns:
            z_quantized: quantized patches from the codebook
            quantize_err: quantization L2 error
            encoding_indices: indices of the closest codebook word for each patch
        """
        # reshape z -> (batch, height, width, channel) and flatten
        # z, 'b c h w -> b h w c'
        z = rearrange(z, 'b c h w -> b h w c')
        z = l2norm(z)
        z_flattened = z.reshape(-1, self.cfg.emb_dim)
        self.embedder.init_embed_(z_flattened)

        # l2 distance between normalized vectors z_flattened and codebook words
        dist_l2 = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedder.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedder.weight)  # 'n dist_l2 -> dist_l2 n'

        # indices of codebook 1-NN elements of input vectors z
        encoding_indices = torch.argmin(dist_l2, dim=1)

        # match each vector z with its corresponding codebook word
        z_quantized = self.embedder(encoding_indices).view(z.shape)

        # one-hot encoding of the encoding indices into a codebook
        encodings_onehot = F.one_hot(encoding_indices, self.cfg.num_tokens).type(z.dtype)
        encoding_bins = encodings_onehot.mean(1)
        encoding_indices = encoding_indices.view(z.shape[:-1])
        encoding_bins=encoding_bins.view(z.shape[:-1])
        if not self.training:
            with torch.no_grad():
                cluster_size = encodings_onehot.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.cfg.decay_codebook_mov_avg)

        if self.training and self.embedder.update:
            # EMA cluster size

            batch_bins = encodings_onehot.sum(0)
            self.all_reduce_fn(batch_bins)

            # self.embedding.cluster_size_ema_update(bins)
            ema_inplace(self.cluster_size, batch_bins, self.cfg.decay_codebook_mov_avg)

            zero_mask = (batch_bins == 0)
            batch_bins = batch_bins.masked_fill(zero_mask, 1.)

            embed_sum = z_flattened.t() @ encodings_onehot
            self.all_reduce_fn(embed_sum)

            embed_normalized = (embed_sum / batch_bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(zero_mask[..., None], self.embedder.weight,
                                           embed_normalized)

            norm_ema_inplace(self.embedder.weight, embed_normalized, self.cfg.decay_codebook_mov_avg)

        # compute quantize_err for embedding = 1 - cosine similarity between z_quantized and z_e
        quantize_err = self.cfg.betta_quantize_err * F.mse_loss(z_quantized.detach(), z)

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match the original input shape
        # z_quantized, 'b h w c -> b c h w'
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w')
        return z_quantized, quantize_err, encoding_indices, encoding_bins

    def reset_cluster_size(self, device):
        if self.cfg.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.cfg.num_tokens))
            self.cluster_size = self.cluster_size.to(device)

class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5, kmeans_init=True, codebook_init_path: str = ''):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps
        if codebook_init_path == '':
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = l2norm(weight)
            else:
                weight = torch.zeros(num_tokens, codebook_dim)
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))

        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        # self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        print("Performing Kemans init for codebook")
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        # normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        # embed_normalized = l2norm(self.embed_avg / smoothed_cluster_size.unsqueeze(1))
        self.weight.data.copy_(embed_normalized)


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


def norm_ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
    moving_avg.data.copy_(l2norm(moving_avg.data))
