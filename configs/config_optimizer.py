import enum
from dataclasses import dataclass, field
from typing import Optional, List

from configs.config_base import ParamsBase


class OptimizerTypes(str, enum.Enum):
    ADAM = 'adam'
    ADAMW = 'adamw'
    SGD = 'sgd'

@dataclass
class ConfigClassifierOptimizer(ParamsBase):
    lr: float = 5e-4
    min_lr: float = 1e-5
    warmup_lr: float = 1e-6
    layer_decay: float = 0.9  # Layer-wise learning rate decay
    warmup_epochs: int = 5  # epochs to warmup LR, if scheduler supports
    warmup_steps: int = -1  # num of steps to warmup LR, will overload warmup_epochs if set > 0
    eps: float = 1e-8 # Optimizer Epsilon
    betas: Optional[List[float]] = None # field(default_factory=lambda: [0.9, 0.999])
    clip_grad: Optional[float] = None # Clip gradient norm
    skip_weight_decay_list: List[str] = field(default_factory=lambda: [])
    weight_decay: float  = 0.05 # weight decay
    # """Final value of the
    #         weight decay. We use a cosine schedule for WD and using a larger decay by
    #         the end of training improves performance for ViTs."""
    weight_decay_end: Optional[float] = None
    optimizer_type: str = OptimizerTypes.ADAMW.value
    momentum: float = 0.9 # SGD momentum
    weight_init: float = 0.001
    # name of layers for extract from optimi
    filter_layers: List[str] = field(default_factory=lambda: [])
    disable_weight_decay_on_rel_pos_bias: bool = False

    def __post_init__(self):
        pass
        # if isinstance(self.optimizer_type, str):
        #     self.optimizer_type = OptimizerTypes(self.optimizer_type)

    def check_valid(self):
        pass