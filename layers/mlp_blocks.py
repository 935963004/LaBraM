from typing import Callable
from timm.layers import trunc_normal_
from torch import nn as nn


class MlpClassifier(nn.Module):
    def __init__(self,
                 in_features: int,
                 act_layer: Callable = nn.GELU,
                 num_classes: int = 1,
                 depth: int = 3,
                 drop: float =0.,
                 layer_ratio: float=0.5):
        """Initializes MLP classifier with fully connected blocks"""
        super().__init__()
        self.blocks_fc = []
        in_dim = in_features
        for _ in range(depth):
            out_dim = int(in_dim * layer_ratio)
            self.blocks_fc.append(FcBlock(in_dim, out_dim, act_layer=act_layer, drop=drop))
            in_dim = out_dim
        self.blocks_fc = nn.ModuleList(self.blocks_fc)
        self.fc_out = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        for fc in self.blocks_fc:
            x = fc(x)
        return self.fc_out(x)


class FcBlock(nn.Module):
    def __init__(self, in_features, out_features, act_layer=nn.GELU, drop=0.):
        """Initializes fully-connected block with activation and dropout"""
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()
        trunc_normal_(self.fc.weight, std=.02)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """Initializes MLP model with configurable dimensions and dropout"""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x

