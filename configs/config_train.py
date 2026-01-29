from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from configs.config_base import ParamsBase

@dataclass
class ConfigClassifierTrainer(ParamsBase):
    def check_valid(self):
        pass