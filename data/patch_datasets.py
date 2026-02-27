import os
import pickle
from _warnings import warn
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from einops import rearrange
from scipy.signal import resample
from sklearn.model_selection import train_test_split, StratifiedKFold

from configs import ConfigProcEEGDataset
from configs.config_data import ConfigProcEEGDataset
from data.hdf5_datasets import ShockDataset
from data.eeg_consts import MIN_DATA_LENGTH, DEFAULT_SAMPLING_RATE, DEFAULT_SEGMENT_LENGTH_SECS
from configs.serialization import to_data_file, from_data_file

class HMCLoader(Dataset):
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False,
                 is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        if is_instruct:
            # TODO:  fix this logic
            raise NotImplementedError("free text instruction not supported yet")
            # enc = tiktoken.get_encoding("gpt2")
            # encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # # 50257 for [SEP]
            # self.text = {
            #     0: torch.IntTensor([50257] + encode(
            #         'Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (A) <|endoftext|>')),
            #     1: torch.IntTensor([50257] + encode(
            #         'Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (B) <|endoftext|>')),
            #     2: torch.IntTensor([50257] + encode(
            #         'Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (C) <|endoftext|>')),
            #     3: torch.IntTensor([50257] + encode(
            #         'Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (D) <|endoftext|>')),
            #     4: torch.IntTensor([50257] + encode(
            #         'Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (E) <|endoftext|>')),
            # }
            # self.prompt = torch.IntTensor([50257] + encode(
            #     'Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: ('))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = int(sample["y"])

        data = torch.FloatTensor(X / 100)
        # time = data.size(1) // 200
        # input_time = [i for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        return data, Y
        ch_names = sample["ch_names"]
        # input_chans = list(ch_names) * time
        #
        # if not self.is_instruct:
        #     input_chans = torch.IntTensor(get_chans(input_chans))
        #     input_time = torch.IntTensor(input_time)
        #
        #     gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
        #     num_chans = len(ch_names)
        #     for i in range(time):
        #         gpt_mask[:, i * num_chans:(i + 1) * num_chans, i * num_chans:(i + 1) * num_chans] = 1
        #     return data, Y #, input_chans, input_time, gpt_mask.bool()
        # else:
        #     pass
        #
        # if self.is_val:
        #     text = self.prompt
        # else:
        #     text = self.text[int(Y)]
        #     # pad text to text_max_len
        #     valid_text_len = text.size(0)
        #     if self.text_max_len > valid_text_len:
        #         text_pad = torch.full((self.text_max_len,), fill_value=50256)
        #         text_pad[:valid_text_len] = text
        #         text = text_pad
        #
        # # pad eeg to eeg_max_len
        # valid_eeg_len = data.size(0)
        # if self.eeg_max_len > data.size(0):
        #     X_eeg = torch.zeros((self.eeg_max_len, 200))
        #     X_eeg[:data.size(0)] = data
        #     eeg_mask = torch.ones(self.eeg_max_len)
        #     eeg_mask[valid_eeg_len:] = 0
        #
        #     input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
        #     input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        # else:
        #     X_eeg = data
        #     eeg_mask = torch.ones(data.size(0))
        #
        # input_chans = torch.IntTensor(get_chans(input_chans))
        # input_time = torch.IntTensor(input_time)
        #
        # num_tokens = X_eeg.size(0) + text.size(0)
        # gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        # num_chans = len(ch_names)
        # for i in range(time):
        #     gpt_mask[:, i * num_chans:(i + 1) * num_chans, i * num_chans:(i + 1) * num_chans] = 1
        # gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        #
        # if self.is_val:
        #     return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        #
        # Y_text = torch.full_like(text, fill_value=-1)
        # prompt_len = self.prompt.size(0) - 1
        # Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        # return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()


class InternalDataset(Dataset):
    def __init__(self,
                 cfg: ConfigProcEEGDataset,
                 # ds_path: str,
                 # metadata_csv_path: str,
                 # class_labels: List[str],
                 # is_normal_abnormal: bool = False, ## class_labels[0] must be normal class label
                 len_in_sec: int = DEFAULT_SEGMENT_LENGTH_SECS,
                 is_random: bool = False):
        """
        Initializes an instance of PyTorch Dataset that loads prossesced patches of Inetral EEG data
        with corresponding metadata from a CSV file and return item for training with requred labels

        Parameters:
            ds_path (str): The directory path to the dataset containing prossed EEG signals
            that split to segments from `.npy` files.
            metadata_csv_path (str): The file path to the metadata CSV containing information about the EEG data.
            class_labels (list): A list of column names in the CSV file representing class label indicators.
            is_normal_abnormal (bool): A flag indicating if the labels should be processed as binary
                                        "normal vs abnormal". Defaults to False. If True, `class_labels[0]`
                                        must correspond to the 'normal' class label.
            len_in_sec (int): The desired input segment length in seconds.
                                Defaults to 10 seconds (split to 10 patches). Used to calculate
                              the number of required sampling points per segment.
            is_random (bool): Whether to shuffle/select segments randomly for loading. Defaults to False.

        """
        self.ds_path = cfg.dataset_path
        self.is_normal_abnormal = cfg.is_normal_abnormal
        class_labels = cfg.label_names
        self.metadata_df: pd.DataFrame = pd.read_csv(cfg.metadata_csv_path)
        if not set(class_labels).issubset(self.metadata_df.columns):
            raise ValueError(f"Metadata CSV is missing required columns: {set(class_labels) - set(self.metadata_df.columns)}")
        self.metadata_df['class_label'] = np.argmax(self.metadata_df[class_labels], 1)
        if self.is_normal_abnormal:
            self.metadata_df['class_label'] = self.metadata_df['class_label'] == 0
        else:
            self.metadata_df = self.metadata_df[(self.metadata_df[class_labels].sum(1) == 1.0)]

        eeg_np_files = list(Path(self.ds_path).glob("*.npy"))
        id_keys = list(map(lambda x: x.stem.split("_")[0], eeg_np_files))
        id_interval = list(map(lambda x: x.stem.split("_")[1], eeg_np_files))
        ids_unique = np.unique(id_keys, return_counts=False)
        self.eeg_files_df: pd.DataFrame = pd.DataFrame({"id_key": id_keys,
                                                        "eeg_np_file": eeg_np_files,
                                                        "id_interval": id_interval}).set_index("id_key")
        self.metadata_df['id_key'] = self.metadata_df['filename_hashed'].apply(lambda x: x.split("_")[0])
        self.metadata_df = self.metadata_df.set_index("id_key")
        ids_common = self.metadata_df.index.intersection(ids_unique)
        self.metadata_df = self.metadata_df.loc[ids_common]
        self.eeg_files_df = self.eeg_files_df.loc[ids_common]

        if self.metadata_df.shape[0] < MIN_DATA_LENGTH:
            raise ValueError(f"Metadata CSV contains less than {MIN_DATA_LENGTH} rows.")


        self.default_rate = DEFAULT_SAMPLING_RATE

        self.len_sampling = len_in_sec * DEFAULT_SAMPLING_RATE
        self.is_random = is_random
        self.class_labels = class_labels
        self.n_classes = len(self.class_labels)

    def __len__(self) -> int:
        return self.eeg_files_df.shape[0]

    # List of unique identifiers of EEG samples
    @property
    def id_keys(self)->List[str]:
        return self.metadata_df.index.tolist()

    def get_subset(self, id_keys: List[str]) -> torch.utils.data.Subset:
        """
        Create a subset of the current dataset based on specified keys of EEG samples.
        Arguments:
        id_keys: List[str]
            A list of identifier keys of EEG samples .
        Returns:
        torch.utils.data.Subset
            A subset of the dataset containing only the entries that match the specified identifier keys.
        """
        file_ids = np.nonzero(self.eeg_files_df.index.isin(id_keys))[0]
        return torch.utils.data.Subset(self, file_ids)

    def __getitem__(self, ind: Union[int, List[int]]) -> Dict[str, torch.Tensor]:
        """
        Retrieves an item of EEG data and corresponding metadata based on the provided index( or multi index N/A now).



        Parameters:
        ind (Union[int, List[int]]): Index or indices specifying the EEG data to retrieve.

        Returns:
        Dict[str, torch.Tensor]: A dictionary of data item containing the following keys:
          - 'data': Processed asegment EEG file data as a torch.FloatTensor.
          - 'label': Class/classes label associated with the EEG data.
          - 'id_key': Unique identifier key corresponding to the retrieved data.
          - 'id_interval': Interval identifier for the EEG data.

        Raises:
        NotImplementedError: Raised when specific behaviors for custom indices or random sampling
                             are not yet implemented.

        Warns:
        UserWarning: Issued if the length of the EEG data exceeds the predefined sampling length.
        """
        file_info = self.eeg_files_df.iloc[ind]
        id_key = file_info.name
        id_interval = file_info['id_interval']
        file_eeg_path = file_info['eeg_np_file']
        row_ind = self.metadata_df.loc[id_key]
        data_eeg = np.load(file_eeg_path)
        if self.is_random:
            start_idx = np.random.randint(0, data_eeg.shape[0] - self.len_sampling)
            raise NotImplementedError
        elif isinstance(ind, list):
            start_idx = ind[1]
            raise NotImplementedError
        else:
            start_idx = 0
        if data_eeg.shape[1] > self.len_sampling:
            len_sec = data_eeg.shape[1] / DEFAULT_SAMPLING_RATE
            warn(f"EEG file {file_eeg_path} is longer than {len_sec} seconds.")
            data_eeg = data_eeg[:, start_idx:start_idx + self.len_sampling]
        data_eeg = torch.FloatTensor(data_eeg)
        label  = row_ind['class_label'].astype(np.int_)
        item_data = {'data': data_eeg,
                     'label': label,
                     'id_key': id_key,
                     'id_interval': id_interval,}
        return item_data


class TUABLoader(Dataset):
    def __init__(self, root, files, sampling_rate=DEFAULT_SAMPLING_RATE):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class TUEVLoader(Dataset):
    """
    A PyTorch dataset that load prosessed to patches from TUEV dataset files.



    Attributes:
        root (str): The root directory path where the data files are stored.
        files (List[str]): A list of file names within the root directory
            to be loaded and managed.
        default_rate (int): The default sampling rate for the dataset
            (default value is 200).
        sampling_rate (int): The user-specified sampling rate for the
            dataset. Defaults to 200 if not provided.
    """
    def __init__(self, root: str, files: List[str], sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y


def prepare_HMC_dataset(root: Path, is_instruct: bool = False, eeg_max_len: int = -1, text_max_len: int = -1)->Tuple[Dataset, Dataset, Dataset]:
    train_files = os.listdir(os.path.join(root, "../train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = HMCLoader(os.path.join(root, "../train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = HMCLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = HMCLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUEV_dataset(root)->Tuple[Dataset, Dataset, Dataset]:
    # set random seed
    seed = 4523
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test data loader
    train_dataset = TUEVLoader(
        os.path.join(
            root, "processed_train"), train_files
    )
    test_dataset = TUEVLoader(
        os.path.join(
            root, "processed_test"), test_files
    )
    val_dataset = TUEVLoader(
        os.path.join(
            root, "processed_eval"), val_files
    )
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUAB_dataset(root)->Tuple[Dataset, Dataset, Dataset]:
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "../train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "../train"), train_files)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_internal_dataset(cfg: ConfigProcEEGDataset,
work_dir
        # ds_path: Union[str, Path],
        #                      class_labels: List[str],
        #                      is_normal_abnormal: bool = False,
        #                      metadata_csv_path: Optional[str] = None,
        #                      data_split: List[float]=None,
        #                      seed: int =4523
                             ) -> Tuple[Dataset, Dataset, Dataset]:
    """Prepares stratified train/validation/test splits from internal dataset"""
    if cfg.data_split is None or len(cfg.data_split) != 3:
        cfg.data_split = [0.8, 0.1, 0.1]
    assert sum(cfg.data_split) == 1.0, \
        "data_split must sum to 1.0"
    assert len(cfg.data_split) == 3, \
        "data_split must have 3 elements: train, val, test"

    cfg.metadata_csv_path = os.path.join(cfg.dataset_path, "metadata.csv") if cfg.metadata_csv_path is None else cfg.metadata_csv_path
    assert os.path.isfile(cfg.metadata_csv_path), \
        f"metadata_csv_path {cfg.metadata_csv_path} does not exist"
    assert os.path.isdir(cfg.dataset_path), \
        f"root_path {cfg.dataset_path} is not a directory"
    eeg_dataset = InternalDataset(cfg)
    assert len(eeg_dataset) > MIN_DATA_LENGTH, \
        f"No data found in {cfg.dataset_path}"
    if cfg.fold_split_path is None:
        data_split_ids = split_data_ids(eeg_dataset, cfg)

        cfg.fold_split_path  = os.path.join(work_dir, 'fold_split_ids.yaml')
        to_data_file(data_split_ids, cfg.fold_split_path )
    else:
        print(f"Loading fold_split_path from {cfg.fold_split_path}")
        data_split_ids = from_data_file(cfg.fold_split_path)

    train_id = data_split_ids['train']
    valid_id = data_split_ids['valid']
    test_id = data_split_ids['test']
    train_dataset = eeg_dataset.get_subset(train_id)
    valid_dataset = eeg_dataset.get_subset(valid_id)
    test_dataset = eeg_dataset.get_subset(test_id) if test_id is not None else None
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(eeg_dataset,
    #                                                              data_split,
    #                                                              generator=split_generator)
    if len(train_dataset) < MIN_DATA_LENGTH:
        raise RuntimeError(f"No data found in train_dataset")

    if len(valid_dataset) < MIN_DATA_LENGTH:
        raise RuntimeError(f"No data found in val_dataset")

    if test_dataset is not None and len(test_dataset) < MIN_DATA_LENGTH:
        raise RuntimeError(f"No data found in test_dataset")

    return train_dataset, valid_dataset, test_dataset

def cross_validation_split_data_ids(eeg_dataset: InternalDataset,
                                    cfg: ConfigProcEEGDataset) -> List[Dict[str, List[str]]]:
    dataset_ids = eeg_dataset.id_keys
    skf = StratifiedKFold(n_splits=cfg.cross_valid_folds,
                          shuffle=True,
                          random_state=int(cfg.seed_ds_split_train))

    splits: List[Dict[str, List[str]]] = []
    y = np.zeros(len(dataset_ids), dtype=np.int64)
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(dataset_ids)), y)):
        train_ids = [dataset_ids[i] for i in train_idx.tolist()]
        valid_ids = [dataset_ids[i] for i in valid_idx.tolist()]
        split_fold = {'train': train_ids, 'valid': valid_ids}
        splits.append(split_fold)
    return splits


def split_data_ids(eeg_dataset: InternalDataset, cfg: ConfigProcEEGDataset) -> Dict[str, List[str]]:
    train_id, valid_test_id = train_test_split(eeg_dataset.id_keys,
                                               train_size=cfg.data_split[0],
                                               random_state=cfg.seed_ds_split_train)
    if cfg.data_split[2] >0:
        valid_id, test_id = train_test_split(valid_test_id,
                                             train_size=cfg.data_split[2] / (cfg.data_split[1] + cfg.data_split[2]),
                                             random_state=cfg.seed_ds_split_train)
        data_split_ids = {'train': train_id, 'valid': valid_id, 'test': test_id}
    else:
        data_split_ids = {'train': train_id, 'valid': valid_test_id, 'test': None}
    return data_split_ids


def build_pretraining_dataset(datasets: list, time_window: list, stride_size=200, start_percentage=0, end_percentage=1):
    shock_dataset_list = []
    ch_names_list = []
    for dataset_list, window_size in zip(datasets, time_window):
        dataset = ShockDataset([Path(file_path) for file_path in dataset_list], window_size * 200, stride_size, start_percentage, end_percentage)
        shock_dataset_list.append(dataset)
        ch_names_list.append(dataset.get_ch_names())
    return shock_dataset_list, ch_names_list
