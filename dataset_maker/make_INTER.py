import duckdb
import mne
import pandas as pd
import numpy as np
import os
import pickle
from warnings import warn
from sklearn.frozen.tests.test_frozen import regression_dataset
from torch.utils.hipify.hipify_python import meta_data
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from typing import Dict, Any, List, Union, Iterable
from pathlib import Path

from mne.io import  RawArray


CH_DB = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
         'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']

MIN_LEN_SEC = 200
TBL_DB_NAME = "main.results"

def process_inter_dataset():
    meta_data_path = "/home/leong/data/EEG/INTER_DATA/all_labels_int20K_eeg.csv"
    input_path = "/home/leong/data/EEG/INTER_DATA/20K-EEG/"
    out_path = "/home/leong/data/EEG/INTER_DATA/lesion_control_processed_10sec"
    labels = ["is_control", "is_lesion"]
    n_samples = 1000
    balanced = False
    long_sec = 10
    n_jobs = 16#16
    is_save_intervals = False

    config_raw = {"eeg_file_suffix": ".fif",
                  "l_freq": 0.1,
                  "h_freq": 75.0,
                    "sec_sample": 200,
                  "notch_filter_freq": 50.0,
                  "len_seq_sec": long_sec,
                  "meta_data_path": meta_data_path,
                  "labels": labels,
                  "n_jobs": n_jobs,
                  "resample_n_jobs": 1,
                  "units": 'uV'}
    process_all_to_fit(input_path, config_raw, out_path)

def process_all_to_fit_files(db_path: str,
                             config_raw: dict,
                             out_path: str,
                             jobs: int = 10):
    pool = Pool(processes=jobs)
    pool.map(partial(process_duckdb_to_fit,
                     config_raw=config_raw,
                     out_path=out_path),
             [db_path])
    pool.close()
    pool.join()

def process_duckdb_to_fit(db_path: str, config_raw: dict, out_path: str):
    data_df = read_duckdb(db_path)
    id_names = data_df["id"].unique()
    for subject_id in id_names:
        id_df = data_df[data_df["id"] == subject_id]
        raw_data= convert_raw_mne(id_df, subject_id)
        raw_data = process_raw(raw_data, config_raw)
        out_file = os.path.join(out_path, f"{subject_id}_raw.fif")
        raw_data.save(out_file, overwrite=True)

def read_eeg_channels_from_raw_mne(file_path: Union[str, Path], eeg_channels=None) -> RawArray:
    if eeg_channels is None:
        eeg_channels = CH_DB
    raw_data = mne.io.read_raw(file_path, preload=True)
    raw_data = raw_data.rename_channels(lambda x: x.upper())
    raw_data = raw_data.pick_channels(eeg_channels).reorder_channels(eeg_channels)

    return raw_data

def read_duckdb(path) -> pd.DataFrame:
    con_db= duckdb.connect(path)
    return con_db.execute(f"SELECT * FROM {TBL_DB_NAME}").fetchdf()

def convert_raw_mne(subject_df: pd.DataFrame, subject_id: str) -> RawArray:
    subject_df = subject_df.sort_values(by='time')
    s_freq = int(np.round(1 / subject_df['time'].diff().mean()))
    data_array = np.array(subject_df[CH_DB], dtype=np.float64)
    info_id = mne.create_info(ch_names=CH_DB, sfreq=s_freq, ch_types='eeg')
    raw_array = mne.io.RawArray(data_array.T, info_id)
    if raw_array.duration < MIN_LEN_SEC:
        raise ValueError(f"Subject {subject_id} has too short duration: {raw_array.duration} sec < {MIN_LEN_SEC} sec")
    return raw_array

def process_raw(raw: RawArray, config: Dict[str, Any], n_jobs: int = 1) -> RawArray:
    l_freq = config["l_freq"]
    h_freq = config["h_freq"]
    notch_filter_freq = config["notch_filter_freq"]
    sec_sample = config["sec_sample"]
    ch_names = raw.ch_names
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=ch_names, n_jobs=n_jobs, verbose=False)
    raw.notch_filter(notch_filter_freq, picks=ch_names, n_jobs=n_jobs)
    raw.resample(sec_sample, n_jobs=n_jobs)

    return raw

def process_eeg_files(file_paths: Iterable[Union[str,Path]],
                      config: Dict[str, Any],
                      out_path: str) -> List[str]:
    file_paths = list(file_paths)
    if len(file_paths) == 0:
        raise ValueError("No files to process")
    assert all(map(lambda x: x.is_file(), file_paths)), "Not all files exist"
    out_path = Path(out_path)
    assert out_path.is_dir(), f"Output path {out_path} does not exist"
    len_seq_sec = config["len_seq_sec"]
    len_samples = len_seq_sec * config["sec_sample"]
    resample_n_jobs = config["resample_n_jobs"]
    out_files = []
    eeg_channels = CH_DB
    channel_type_mapping = {ch_name: "eeg" for ch_name in eeg_channels}
    for file_path in tqdm(file_paths):
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileExistsError(f"File {file_path} does not exist")
        id_key = file_path.stem.split("_")[0]
        raw_array = read_eeg_channels_from_raw_mne(file_path, eeg_channels=eeg_channels)
        if raw_array.duration < MIN_LEN_SEC:
            warn(f"Subject {id_key} has too short duration: {raw_array.duration} sec < {MIN_LEN_SEC} sec")
            continue
        # raw_array = raw_array.crop(0, len_seq_sec)
        raw_array = process_raw(raw_array, config, n_jobs=resample_n_jobs)
        raw_array = raw_array.set_channel_types(channel_type_mapping,
                                                on_unit_change="ignore",
                                                verbose=False)


        if config.get("save_intervals", False):
            eeg_array = raw_array.get_data(units=config["units"], picks=eeg_channels)
            save_chunks_files = save_eeg_intervals(eeg_array, out_path, id_key, len_samples)
        else:
            out_file = Path(out_path, id_key).with_suffix(".fif")
            raw_array.save(out_file, overwrite=True)

        # if np.abs(eeg_array.shape[1] - len_seq_sec * config["sec_sample"]) > 1.0:
        #     raise ValueError(f"Raw array shape {eeg_array.shape[1] / config['sec_sample']} "
        #                      f"does not match expected length {len_seq_sec}")
        # out_file = Path(out_path, id_key).with_suffix(".npy")
        # np.save(out_file, eeg_array)
        # if not out_file.exists():
        #     raise FileExistsError(f"File {out_file} does not exist")
        # raw_array.save(out_file, overwrite=True)
        out_files += save_chunks_files
    return out_files

def save_eeg_intervals(eeg_array: np.ndarray, save_path: Union[str, Path], id_key: str, len_samples: int) -> List[Path]:
    n_chunks = eeg_array.shape[1] // len_samples
    eeg_chunks = np.hsplit(eeg_array[:,:n_chunks* len_samples], n_chunks)
    out_files = []
    for ind in range(n_chunks):
        out_file = Path(save_path, f"{id_key}_{ind}").with_suffix(".npy")
        np.save(out_file, eeg_chunks[ind])
        out_files.append(out_file)
    return out_files

def process_all_to_fit(input_path: str, config: Dict[str, Any], out_path: str):
    os.makedirs(out_path, exist_ok=True)
    input_path = Path(input_path)
    assert input_path.is_dir(), f"Input path {input_path} does not exist"
    pattern = "*"+config["eeg_file_suffix"]
    file_eeg_paths = list(input_path.glob(pattern))

    assert len(list(file_eeg_paths)) > 0, f"No eeg files found in {input_path}"

    assert all(map(lambda x: x.is_file(), file_eeg_paths)), \
        f'Not all files are eeg files{pattern} in path: {input_path.name}'

    # Filter relevant files:
    if "labels" in config and len(config["labels"]) > 0:
        meta_data_df = pd.read_csv(config["meta_data_path"])
        labels = config["labels"]
        if not set(labels) < set(meta_data_df.columns):
            raise ValueError(f"Labels {labels} not found in meta data columns {meta_data_df.columns}")
        file_names_labels = meta_data_df[meta_data_df[labels].any(axis=1)]["fif_filename_hashed"]
        file_eeg_paths = list(filter(lambda x: x.name in list(file_names_labels), file_eeg_paths))
        if len(file_eeg_paths) == 0:
            raise ValueError(f"No files left after filtering by labels {labels}")
    n_jobs = config.get("n_jobs")
    if n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            data_chunks = np.array_split(file_eeg_paths, n_jobs)
            pool.map(partial(process_eeg_files, config=config, out_path=out_path), data_chunks)
    else:
        process_eeg_files(file_eeg_paths, config, out_path)

if __name__ == "__main__":
    """
    """
    process_inter_dataset()




