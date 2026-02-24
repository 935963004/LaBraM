from typing import Optional, Union
from torch import nn
from configs import NormLayer

def get_norm_layer(norm_layer: Union[None, str, NormLayer], dim: Optional[int] = None, **kwargs) -> nn.Module:
    """ Factory method for normalization layers"""
    if norm_layer is None:
        return nn.Identity()
    norm_layer = NormLayer(norm_layer)
    if norm_layer == NormLayer.LAYER_NORM:
        return nn.LayerNorm(dim,**kwargs)
    else:
        raise ValueError(f"Unsupported norm layer: {norm_layer}")

