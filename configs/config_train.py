from dataclasses import dataclass, field
from typing import Optional

from configs.config_base import ParamsBase


@dataclass
class LossesWeightsConfig(ParamsBase):
    classifier: float = 1.0
    phase_recon: float = 0.1
    magnitude_recon: float = 1.0
    frequency_cutoff: float = 1.0
    quantize_err: float = 1.0

    def check_valid(self):
        pass


@dataclass
class ConfigClassifierTrainer(ParamsBase):
    epochs: int = 30
    ckpt_encoder_path: Optional[str] = None
    ckpt_vqnsp_path: Optional[str] = None
    device: str = 'cuda'
    seed: int = 1984
    losses_weights: LossesWeightsConfig = field(default_factory=lambda: LossesWeightsConfig())
    label_smoothing: float = 0.1  # Label smoothing in Cross entropy loss
    auto_resume: bool = True
    resume_ckpt_path: Optional[str] = None
    start_epoch: int = 0
    num_workers: int = 16
    pin_mem: bool = True
    dist_eval: bool = False
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_force_cpu: bool = False
    enable_deepspeed: bool = False
    distributed: bool = False

    def check_valid(self):
        pass