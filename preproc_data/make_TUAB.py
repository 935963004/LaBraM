# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BIOT code base
# https://github.com/ycq091044/BIOT
# --------------------------------------------------------
import os
import pickle
from functools import partial
from multiprocessing import Pool
import numpy as np
import mne


drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

standard_channels = [
    "EEG FP1-REF",
    "EEG F7-REF",
    "EEG T3-REF",
    "EEG T5-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F8-REF",
    "EEG T4-REF",
    "EEG T6-REF",
    "EEG O2-REF",
    "EEG FP1-REF",
    "EEG F3-REF",
    "EEG C3-REF",
    "EEG P3-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F4-REF",
    "EEG C4-REF",
    "EEG P4-REF",
    "EEG O2-REF",
]


def split_and_dump(data_chunk, config: dict = None):
    fetch_folder, sub, dump_folder, label = data_chunk
    # config = {"l_freq": 0.1, "h_freq": 75.0, "sec_sample": 200, "notch_filter_freq": 50.0,
    #           "len_seq_sec": 10, "n_jobs": 5}
    l_freq = config["l_freq"]
    h_freq = config["h_freq"]
    sec_sample = config["sec_sample"]
    notch_filter_freq = config["notch_filter_freq"]
    len_seq_sec = config["len_seq_sec"]
    n_jobs_sample = config["n_jobs"]
    units = config["units"]
    seq_len = int(len_seq_sec * sec_sample)


    for file in os.listdir(fetch_folder):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            try:
                if drop_channels is not None:
                    useless_chs = []
                    for ch in drop_channels:
                        if ch in raw.ch_names:
                            useless_chs.append(ch)
                    raw.drop_channels(useless_chs)
                if chOrder_standard is not None and len(chOrder_standard) == len(raw.ch_names):
                    raw.reorder_channels(chOrder_standard)
                if raw.ch_names != chOrder_standard:
                    raise Exception("channel order is wrong!")

                raw.filter(l_freq=l_freq, h_freq=h_freq)
                raw.notch_filter(notch_filter_freq)
                raw.resample(sec_sample, n_jobs=n_jobs_sample)

                ch_name = raw.ch_names
                raw_data = raw.get_data(units=units)
                channeled_data = raw_data.copy()
            except:
                with open("tuab-process-error-files.txt", "a") as f:
                    f.write(file + "\n")
                continue
            for i in range(channeled_data.shape[1] // seq_len):
                dump_path = os.path.join(
                    dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
                )
                pickle.dump(
                    {"X": channeled_data[:, i * seq_len : (i + 1) * seq_len], "y": label},
                    open(dump_path, "wb"),
                )


if __name__ == "__main__":
    """
    TUAB dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    """
    # root to abnormal dataset
    root = "/Users/leon/Data/EEG-public/TAUB/TUH_Abnormal/v3.0.0/edf/"
    channel_std = "01_tcp_ar"
    out_name = "processed"
    ## l_freq=0.1, h_freq=75.0
    config = {"l_freq": 0.1,
              "h_freq": 75.0,
              "sec_sample": 200,
              "notch_filter_freq": 50.0,
              "len_seq_sec": 10,
              "n_jobs": 5,
              "units": 'uV'}

    n_jobs = 24
    out_dir = os.path.join(root, out_name)

    # train, val abnormal subjects
    train_val_abnormal = os.path.join(root, "train", "abnormal", channel_std)
    train_val_a_sub = list(
        set([item.split("_")[0] for item in os.listdir(train_val_abnormal)])
    )
    np.random.shuffle(train_val_a_sub)
    train_a_sub, val_a_sub = (
        train_val_a_sub[: int(len(train_val_a_sub) * 0.8)],
        train_val_a_sub[int(len(train_val_a_sub) * 0.8) :],
    )

    # train, val normal subjects
    train_val_normal = os.path.join(root, "train", "normal", channel_std)
    train_val_n_sub = list(
        set([item.split("_")[0] for item in os.listdir(train_val_normal)])
    )
    np.random.shuffle(train_val_n_sub)
    train_n_sub, val_n_sub = (
        train_val_n_sub[: int(len(train_val_n_sub) * 0.8)],
        train_val_n_sub[int(len(train_val_n_sub) * 0.8) :],
    )

    # test abnormal subjects
    test_abnormal = os.path.join(root, "eval", "abnormal", channel_std)
    test_a_sub = list(set([item.split("_")[0] for item in os.listdir(test_abnormal)]))

    # test normal subjects
    test_normal = os.path.join(root, "eval", "normal", channel_std)
    test_n_sub = list(set([item.split("_")[0] for item in os.listdir(test_normal)]))

    # create the train, val, test sample folder
    os.makedirs(out_dir, exist_ok=True)

    # if not os.path.exists(os.path.join(root, "processed", "train")):
    #     os.makedirs(os.path.join(root, "processed", "train"))
    train_dump_folder = os.path.join(out_dir, "train")
    os.makedirs(train_dump_folder, exist_ok=True)

    # if not os.path.exists(os.path.join(root, "processed", "val")):
    #     os.makedirs(os.path.join(root, "processed", "val"))
    val_dump_folder = os.path.join(out_dir, "val")
    os.makedirs(val_dump_folder, exist_ok=True)
    #
    # if not os.path.exists(os.path.join(root, "processed", "test")):
    #     os.makedirs(os.path.join(root, "processed", "test"))
    test_dump_folder = os.path.join(out_dir, "test")
    os.makedirs(test_dump_folder, exist_ok=True)

    # fetch_folder, sub, dump_folder, labels
    #fetch_folder, sub, dump_folder, label = params
    data_chunks = []
    for train_sub in train_a_sub:
        data_chunks.append([train_val_abnormal, train_sub, train_dump_folder, 1])
    for train_sub in train_n_sub:
        data_chunks.append([train_val_normal, train_sub, train_dump_folder, 0])
    for val_sub in val_a_sub:
        data_chunks.append([train_val_abnormal, val_sub, val_dump_folder, 1])
    for val_sub in val_n_sub:
        data_chunks.append([train_val_normal, val_sub, val_dump_folder, 0])
    for test_sub in test_a_sub:
        data_chunks.append([test_abnormal, test_sub, test_dump_folder, 1])
    for test_sub in test_n_sub:
        data_chunks.append([test_normal, test_sub, test_dump_folder, 0])


    # split and dump in parallel
    with Pool(processes=n_jobs) as pool:
        # Use the pool.map function to apply the square function to each element in the numbers list
        result = pool.map(partial(split_and_dump, config=config), data_chunks)