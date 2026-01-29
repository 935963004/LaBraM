import warnings
from abc import ABC
from typing import List, Union, Sequence
import os
from dataclasses import dataclass, field
from configs.config_base import ParamsBase
from configs.config_data import ConfigProcEEGDataset
from configs.config_model import ConfigEEGClassifierModel
from configs.config_train import ConfigClassifierTrainer
from configs.config_optimizer import ConfigClassifierOptimizer

@dataclass
class ConfigClassifierModelRun(ParamsBase):
    # TODO: check default_factory (make instance and not just have the class name)
    data: ConfigProcEEGDataset = field(default_factory=ConfigProcEEGDataset())  # Parameters for data loading
    train: ConfigClassifierTrainer = field(default_factory=ConfigClassifierTrainer())  # Parameters for running training process
    optim: ConfigClassifierOptimizer = field(default_factory=ConfigClassifierOptimizer())  # Parameters for model optimization (training)
    model: ConfigEEGClassifierModel = field(default_factory=ConfigEEGClassifierModel())  # Parameters for model architecture details

    classmethod
    def check_valid(self) -> None:
        self.validate_type_fields()
        self.data.check_valid()
        self.model.check_valid()
        self.optim.check_valid()
        self.train.check_valid()