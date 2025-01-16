from pathlib import Path

import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from mne.io import read_raw_edf
from mne.viz import plot_topomap
from mne.channels import make_standard_montage
from mne import create_info

standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


class SingleEDFDataset(Dataset):
    """Process a single EDF file and create a dataset."""

    def __init__(self, file_path, window_size=2.0, step_size=2.0, threshold_std=5, mask_percentage=0.1):
        """
        Initialize the SingleEDFDataset class.

        Parameters:
        - file_path (str): Path to the EDF file.
        - window_size (float): Window size in seconds.
        - step_size (float): Step size between windows in seconds.
        - threshold_std (float): Threshold for clipping based on standard deviation.
        - mask_percentage (float): Percentage of time to mask in the windows.
        """
        self.file_path = file_path
        self.window_size = window_size
        self.step_size = step_size
        self.threshold_std = threshold_std
        self.mask_percentage = mask_percentage

        self.data = None
        self.sfreq = None
        self.dataset = None
        self.channel_names = None
        self.original_data = None

        # fixed list of "important" channels:
        self.important_channels = ['O2', 'O1', 'C3', 'C4', 'P3', 'P4', 'T4', 'T3', 'Fp2', 'Fp1', 'F3', 'F4', 'Cz', 'Fz',
                                   'F8', 'T6'] #, 'F7', 'T5', 'Pz']  # 19

        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """Load the EDF file and preprocess into windows."""
        try:
            raw = read_raw_edf(self.file_path, preload=True, verbose=False)

            raw.rename_channels({ch_name: ch_name.split('-')[0][:3] for ch_name in raw.ch_names if
                                 '-' in ch_name})  # захардкожено, надо будет переделать (3 з максимальная длина названия канала до спецификации на AA, Av и т.д.)

            # Check if all "important" channels are present
            available_channels = set(raw.ch_names)
            if not all(channel in available_channels for channel in self.important_channels):
                raise ValueError("Not all required channels are present in the file.")

            raw.pick_channels(self.important_channels)  # select only the "important" channels

            imp_ch_set = set([x.upper() for x in self.important_channels])
            reordered_channels = []
            for ch_name in standard_1020:
                if ch_name in imp_ch_set:
                    if ch_name == 'Fp2' or ch_name == 'Fp1':
                        ch_name = 'Fp' + ch_name[2]
                    elif ch_name == 'Cz' or ch_name == 'Fz':
                        ch_name = ch_name[0] + 'z'
                    reordered_channels.append(ch_name)
            raw.reorder_channels(reordered_channels)

            # Preprocess data
            raw.filter(0.5, 40)
            raw.resample(200)

            self.sfreq = raw.info['sfreq']
            self.channel_names = raw.ch_names
            self.original_data = raw.get_data(units='uV')

            data_array = self.original_data.copy()  # work on a copy of the original data
            mean = np.mean(data_array)
            std = np.std(data_array)
            data_array = np.clip(data_array, mean - self.threshold_std * std, mean + self.threshold_std * std)

            rms = np.sqrt(np.sum(data_array ** 2))
            data_array = (data_array - np.mean(data_array, axis=1, keepdims=True)) / rms

            n_samples_window = int(self.window_size * self.sfreq)
            n_samples_step = int(self.step_size * self.sfreq)

            windows = [
                data_array[:, start:start + n_samples_window]
                for start in range(0, data_array.shape[1] - n_samples_window + 1, n_samples_step)
            ]

            self.dataset = torch.tensor(np.array(windows))
        except ValueError as ve:
            print(f"Skipping file {self.file_path}: {ve}")
            self.dataset = None
        except Exception as e:
            raise RuntimeError(f"Error loading or preprocessing the EDF file: {e}")

    def __len__(self):
        return len(self.dataset) if self.dataset is not None else 0

    def __getitem__(self, idx):
        if self.dataset is None:
            raise IndexError("Dataset is empty or not processed due to missing required channels.")
        window = self.dataset[idx].clone()
        # mask_start_idx = random.randint(0, int(window.shape[-1] * (1 - self.mask_percentage)))
        # mask_end_idx = mask_start_idx + int(window.shape[-1] * self.mask_percentage)
        # masked_window = window.clone()
        # masked_window[:, mask_start_idx:mask_end_idx] = 0
        # return masked_window, window, [mask_start_idx, mask_end_idx]
        return window

    def get_ch_names(self):
        return [x.upper() for x in self.important_channels]

    @property
    def feature_size(self):
        return self.dataset[0].shape

    def analyze_data(self):
        """Analyze the processed data (PSD and topomap)."""
        if self.dataset is None or self.sfreq is None:
            raise ValueError("Dataset not processed. Call _load_and_preprocess() first.")

        # Reconstruct processed data from windows
        n_windows, n_channels, window_size = self.dataset.shape
        step_size = int(self.step_size * self.sfreq)
        total_samples = (n_windows - 1) * step_size + window_size

        reconstructed_data = np.zeros((n_channels, total_samples))
        weight_array = np.zeros((n_channels, total_samples))

        for i, window in enumerate(self.dataset.numpy()):
            start = i * step_size
            end = start + window_size
            reconstructed_data[:, start:end] += window
            weight_array[:, start:end] += 1

        # Normalize overlapping regions
        reconstructed_data /= np.maximum(weight_array, 1)

        # PSD
        plt.figure(figsize=(20, 15))
        for channel in range(n_channels):
            plt.psd(reconstructed_data[channel], Fs=self.sfreq, label=f"{self.channel_names[channel]}")
        plt.title("Power Spectral Density (processed data)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (dB/Hz)")
        plt.legend(loc="best")
        plt.show()

        # Topomap
        mean_power_per_channel = np.mean(reconstructed_data ** 2, axis=-1)
        info = create_info(self.channel_names, self.sfreq, ch_types='eeg')
        montage = make_standard_montage('standard_1020')
        info.set_montage(montage)

        print("Power distribution over the head (processed data):")
        fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figure size for larger topomap
        plot_topomap(mean_power_per_channel, info, axes=ax, show=True)

    def compare_original_and_processed(self):
        """Compare the PSD and topomap between original and processed data."""
        if self.dataset is None or self.original_data is None:
            raise ValueError("Dataset not processed. Call _load_and_preprocess() first.")

        # Reconstruct processed data from windows
        n_windows, n_channels, window_size = self.dataset.shape
        step_size = int(self.step_size * self.sfreq)
        total_samples = (n_windows - 1) * step_size + window_size

        processed_data = np.zeros((n_channels, total_samples))
        weight_array = np.zeros((n_channels, total_samples))

        for i, window in enumerate(self.dataset.numpy()):
            start = i * step_size
            end = start + window_size
            processed_data[:, start:end] += window
            weight_array[:, start:end] += 1

        # Normalize overlapping regions
        processed_data /= np.maximum(weight_array, 1)

        # PSD comparison
        plt.figure(figsize=(20, 15))
        colors = plt.cm.tab10.colors
        for channel in range(self.original_data.shape[0]):
            color = colors[channel % len(colors)]
            plt.psd(self.original_data[channel], Fs=self.sfreq, label=f"Original {self.channel_names[channel]}",
                    linestyle='--', color=color)
            plt.psd(processed_data[channel], Fs=self.sfreq, label=f"Processed {self.channel_names[channel]}",
                    color=color)
        plt.title("Power Spectral Density comparison")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (dB/Hz)")
        plt.legend(loc="best")
        plt.show()

        # Topomap comparison
        mean_power_original = np.mean(self.original_data ** 2, axis=-1)
        mean_power_processed = np.mean(processed_data ** 2, axis=-1)

        info = create_info(self.channel_names, self.sfreq, ch_types='eeg')
        montage = make_standard_montage('standard_1020')
        info.set_montage(montage)

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        plot_topomap(mean_power_original, info, axes=axes[0], show=False)
        axes[0].set_title("Original data")
        plot_topomap(mean_power_processed, info, axes=axes[1], show=False)
        axes[1].set_title("Processed data")
        plt.show()


class EDFDataset(Dataset):
    """Integrate multiple EDF files into a dataset."""

    def __init__(self, file_paths, window_size=2.0, step_size=2.0, threshold_std=5, mask_percentage=0.1):
        """
        Initialize the EDFDataset class.

        Parameters:
        - file_paths (list of str): List of paths to EDF files.
        - window_size (float): Window size in seconds.
        - step_size (float): Step size between windows in seconds.
        - threshold_std (float): Threshold for clipping based on standard deviation.
        - mask_percentage (float): Percentage of time to mask in the windows.
        """

        ##########################
        self.datasets = [
            SingleEDFDataset(file_path, window_size, step_size, threshold_std, mask_percentage)
            for file_path in file_paths[:10]
        ]
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]
        self.total_length = sum(self.dataset_lengths)
        ##########################
        # zarina changed from this:
        ##########################
        # self.datasets = []
        # for directories in file_paths:
        #     it = iter(Path(directories).iterdir())
        #     for i in range(10):
        #         file_path = next(it)
        #         print(file_path)
        #         if str(file_path).endswith(".edf"):
        #             self.datasets.append(SingleEDFDataset(file_path, window_size, step_size, threshold_std, mask_percentage))
        # self.dataset_lengths = [len(dataset) for dataset in self.datasets]
        # self.total_length = sum(self.dataset_lengths)
        ##########################

    @property
    def feature_size(self):
        return self.datasets[0].feature_size

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_idx = 0
        while idx >= self.dataset_lengths[dataset_idx]:
            idx -= self.dataset_lengths[dataset_idx]
            dataset_idx += 1
        return self.datasets[dataset_idx][idx]

    def free(self):
        pass

    def get_ch_names(self):
        return self.datasets[0].get_ch_names()
    
    
    
    