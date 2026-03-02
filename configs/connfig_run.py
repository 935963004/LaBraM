from dataclasses import dataclass, field
from typing import Optional
from configs.config_base import ParamsBase
from configs.config_data import ConfigProcEEGDataset
from configs.config_model import ConfigEEGClassifier
from configs.config_optimizer import ConfigClassifierOptimizer
from configs.config_train import ConfigClassifierTrainer


@dataclass
class LoggerParams(ParamsBase):
    experiment: str = 'class_finetune'
    log_dir: str = './logs'
    ckpt_dir: str = './checkpoints'
    run_name: Optional[str] = None
    update_freq: int = 1
    save_ckpt_freq: int = 5

    def check_valid(self) -> None:
        pass


# @dataclass
# class ConfigCheckpointRun(ParamsBase):
#     save_ckpt: bool = True

@dataclass
class ConfigRunClassifierModel(ParamsBase):
    # TODO: check default_factory (make instance and not just have the class name)
    data: ConfigProcEEGDataset = field(
        default_factory=lambda: ConfigProcEEGDataset())  # Parameters for data loading
    train: ConfigClassifierTrainer = field(
        default_factory=lambda: ConfigClassifierTrainer())  # Parameters for running training process
    optim: ConfigClassifierOptimizer = field(
        default_factory=lambda: ConfigClassifierOptimizer())  # Parameters for model optimization (training)
    model: ConfigEEGClassifier = field(
        default_factory=lambda: ConfigEEGClassifier())  # Parameters for model architecture details
    log: LoggerParams = field(default_factory=lambda: LoggerParams())

    def check_valid(self) -> None:
        self.validate_type_fields()
        self.data.check_valid()
        self.model.check_valid()
        self.optim.check_valid()
        self.train.check_valid()