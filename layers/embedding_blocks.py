from typing import Callable, Optional
from torch import nn as nn, Tensor


class FeatureEmbedder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 act_layer: Callable=nn.Tanh,
                 reduce_dim: Optional[int]=None,
                 is_linear: bool = False):
        super().__init__()
        if is_linear:
            if in_dim != out_dim:
                self.embedder = nn.Linear(in_dim, out_dim)
            else:
                self.embedder = nn.Identity()
        else:
            self.embedder= nn.Sequential(
                                        nn.Linear(in_dim, in_dim),
                                        act_layer(),
                                        nn.Linear(in_dim, out_dim))
        self.reduce_dim = reduce_dim

    def forward(self, x: Tensor) -> Tensor:
        if self.reduce_dim is not None:
            x = x.mean(dim=self.reduce_dim)
        return self.embedder(x)


class CodeBookBagEmbedder(nn.Module):
    """
    Represents words from a codebook as a bag of embeddings over patches and channels

    Attributes:
        n_codes (int): Number of tokens(codes) in codebook
        out_dim (int): Dimension of the output embeddings
        act_layer (Callable): internal activation function
    Returns:
        Tensor: Bag of embeddings of shape (batch_size, out_dim)
    """
    def __init__(self,
                 n_codes: int = 8192,
                 out_dim: int = 128,
                 act_layer: Callable = nn.Tanh):
        super().__init__()
        # sprase = True doesnt support backward operation
        self.embedder= nn.Sequential(nn.EmbeddingBag(n_codes, out_dim, sparse=False),
                                    act_layer(),
                                    nn.Linear(out_dim, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.embedder(x.flatten(1))
