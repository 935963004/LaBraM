import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from configs.config_base import ParamsBase


@dataclass
class ConfigProcEEGDataset(ParamsBase):
    # validate_on_init = False
    ds_name: Optional[str] = None  # Name of the dataset used for training
    dataset_path: Optional[str] = None
    metadata_csv_path: Optional[str] = None
    label_names: Optional[List[str]] = field(
        default_factory=lambda: [])  # Name of the label column in the
    is_binary_label: bool = False  # if TRUE labels are binary: 0-Normal(Control) or 1-Abnormal
    is_normal_abnormal: bool = False
    normal_labels: Optional[List[str]] = field(default_factory=lambda: ['is_control'])
    abnormal_labels: Optional[List[str]] = field(default_factory=lambda: ['is_abnormal'])
    num_classes: int = 0

    batch_size_train: int = 8  # Batch size for training-data loading
    batch_size_inference: int = 16  # Batch size for inference-data loading
    n_dataloader_workers_train: int = 0  # Number of threads used in dataloader
    cross_valid_folds: int = 5
    tr_part: float = 0.7  # The training cohort is split to train-data and validation-data (this is ‘internal’ validation). This is the fraction of data used for training. The rest will be used for validation.
    tr_all: bool = False  # If True overrides tr_part and all of the training cohort will be used for training. Note that ‘internal‘ validation still occurs but it results would be overfitted (too good of a convergence)
    ds_part: float = 1.0  # Used in debugging only. Fraction of training-data used for training in debugging mode
    data_type: str = 'pretrained-features'  # Features to be used. Currently only 'pretrained-features' exist
    seed_ds_split_train: Optional[int] = 4523  # Seed for the train-validation split
    seed: int = 4523
    data_split: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1]) # split of train/valid/test
    fold_split_path: Optional[
        str] = None  # Path to JSON file specifying the samples partitioning between the train and the validation datasets in the train/valid split.

    @classmethod
    def check_valid(cls) -> None:
        # TODO: refactoring
        if not Path(cls.dataset_path).is_dir():
            warnings.warn(f"Path to EEF files: {cls.dataset_path} does not exist")
        if len(list(Path(cls.dataset_path).iterdir())) == 0:
            warnings.warn(f"Path to EEF files: {cls.dataset_path} is empty")
        if not Path(cls.metadata_csv_path).is_file():
            warnings.warn(f"Path metadat file: {cls.metadata_csv_path} does not exist")
        if cls.ds_name is None:
            warnings.warn("Dataset name is not specified")
