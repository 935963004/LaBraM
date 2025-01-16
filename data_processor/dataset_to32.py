from pathlib import Path

import os
import numpy as np
from glob import glob
import random
import torch
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from torch.utils.data import Dataset

from mne.io import read_raw_edf
from mne.viz import plot_topomap
from mne.channels import make_standard_montage, DigMontage
from mne import create_info

from constants import standard_channel_positions, important_channels 

class EEGInterpolator:
    """Handles EEG channel interpolation."""
    def __init__(self, standard_channel_positions=standard_channel_positions, n_target_channels=32):
        """
        :param standard_channel_positions: Dictionary of standard channel names to 3D positions.
        :param n_target_channels: Number of target channels to interpolate.
        """
        self.standard_channel_positions = standard_channel_positions
        self.n_target_channels = n_target_channels

    def generate_uniform_sphere_points(self):
        """Generates approximately uniformly distributed points on a sphere."""
        indices = np.arange(0, self.n_target_channels, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / self.n_target_channels)  # Latitude
        theta = np.pi * (1 + 5 ** 0.5) * indices  # Longitude

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.stack([x, y, z], axis=1)

    def interpolate(self, raw_data, channel_positions):
        """
        Interpolates EEG signals from source positions to target positions.

        - raw_data (list of float): 2D numpy array (n_source_channels, n_samples) of EEG signals.
        - channel_positions (list of int): numpy array (n_source_channels, 3) of source channel positions.
        Returns: interpolated EEG signals (n_target_channels, n_samples).
        """
        print(f"Input data signals shape {raw_data.shape}")

        target_positions = self.generate_uniform_sphere_points()
        n_samples = raw_data.shape[1]
        interpolated_signals = np.zeros((self.n_target_channels, n_samples))

        if channel_positions.shape[0] != raw_data.shape[0]:
            raise ValueError(
                f"Mismatch between number of channel positions ({channel_positions.shape[0]}) "
                f"and raw data channels ({raw_data.shape[0]})."
            )

        print(f"Interpolating {raw_data.shape[0]} channels to {self.n_target_channels} target points.")
        print(f"Source positions: {channel_positions.shape}, Target positions: {target_positions.shape}")

        for i in range(n_samples):
            rbf = Rbf(
                channel_positions[:, 0],
                channel_positions[:, 1],
                channel_positions[:, 2],
                raw_data[:, i],
                function='linear'
            )
            interpolated_signals[:, i] = rbf(
                target_positions[:, 0],
                target_positions[:, 1],
                target_positions[:, 2]
            )
            
        print(f"Interpolated data signals shape {interpolated_signals.shape}")
        return interpolated_signals

class SingleEDFDataset(Dataset):
    """Process a single EDF file and create a dataset."""
    def __init__(self, file_path, window_size=2.0, step_size=2.0, threshold_std=5, mask_percentage=0.1, l_freq=0.5, h_freq=40.0, rsfreq=256):
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
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.rsfreq = rsfreq

        self.data = None
        self.sfreq = None
        self.dataset = None
        self.channel_names = None
        self.original_data = None
        self.original_data_interpolated = None
        
        self.interpolator = EEGInterpolator()
        self.important_channels = important_channels # fixed list of "important" channels

        self._preprocess()
        
    def _load_and_standartize(self):
        """Load EDF and rename to standard."""
        raw = read_raw_edf(self.file_path, preload=True)
        
        rename_dict = {}
        for ch_name in raw.ch_names:
            for important in self.important_channels:
                if important in ch_name:
                    rename_dict[ch_name] = important
                    break

        raw.rename_channels(rename_dict)
        raw.pick_channels(self.important_channels)
        raw.reorder_channels(self.important_channels)

        # Check if all "important" channels are present
        available_channels = set(raw.ch_names)
        if not all(channel in available_channels for channel in self.important_channels):
                raise ValueError("Not all required channels are present in the file.")
                
        print(f"Channels in current file: {raw.ch_names}")
        
        return raw
        
    
    def _preprocess(self):
        """Preprocess and split into windows."""
        try:
            raw = self._load_and_standartize()

            # Preprocess data
            raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design='firwin')
            raw.notch_filter(50.0)
            raw.resample(self.rsfreq)
        
            self.sfreq = raw.info['sfreq']
            self.channel_names = raw.ch_names
            self.original_data = raw.get_data(units='uV')
            
            # Interpolate to n_target_channels=32 new channels
            channel_positions = np.array([self.interpolator.standard_channel_positions[ch] for ch in raw.ch_names])
            self.original_data_interpolated = self.interpolator.interpolate(self.original_data.copy(), channel_positions)
            data_array = self.original_data_interpolated.copy()
            
            mean = np.mean(data_array)
            std = np.std(data_array)
            data_array = np.clip(data_array, mean - self.threshold_std * std, mean + self.threshold_std * std)

            # rms = np.sqrt(np.sum(data_array**2))
            # data_array = (data_array - np.mean(data_array, axis=1, keepdims=True)) / rms
            # data_array = self.original_data.copy()
            # mean = np.mean(data_array, axis=1, keepdims=True)
            # std = np.std(data_array, axis=1, keepdims=True)
            # data_array = (data_array - mean) / (std + 1e-6)  # Add epsilon to avoid division by zero

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


    def compare_original_and_processed(self):
        """Compare the PSD and topomap between original and processed data."""
        if self.dataset is None or self.original_data is None:
            raise ValueError("Dataset not processed. Call _load_and_preprocess() first.")

        original_interpolated = self.original_data_interpolated

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
        for channel in range(original_interpolated.shape[0]):
            color = colors[channel % len(colors)]
            plt.psd(original_interpolated[channel], Fs=self.sfreq, label=f"Original Interpolated {channel+1}", linestyle='--', color=color)
            plt.psd(processed_data[channel], Fs=self.sfreq, label=f"Processed {channel+1}", marker="^", markevery=10, linestyle='-', color=color)
        plt.title("Power Spectral Density comparison")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (dB/Hz)")
        plt.legend(loc="best")
        plt.show()
        
    def save_to_h5(self, save_path):
        """Save the preprocessed dataset to HDF5 format."""

        with h5py.File(save_path, 'w') as h5file:
            grp = h5file.create_group(self.file_path.stem)
            dset = grp.create_dataset('eeg', data=self.original_data)

            dset.attrs['lFreq'] = self.l_freq
            dset.attrs['hFreq'] = self.h_freq
            dset.attrs['rsFreq'] = self.rsfreq
            dset.attrs['chOrder'] = self.channel_names

#         # Topomap comparison
#         mean_power_original = np.mean(original_interpolated**2, axis=-1)
#         mean_power_processed = np.mean(processed_data**2, axis=-1)

#         # Create a custom montage with the interpolated positions
#         custom_positions = {
#             f"Ch{idx+1}": pos for idx, pos in enumerate(self.interpolator.generate_uniform_sphere_points())
#         }
#         ch_names = list(custom_positions.keys())
#         pos = np.array(list(custom_positions.values()))

#         montage = DigMontage(ch_pos=dict(zip(ch_names, pos)))

#         info = create_info(ch_names, self.sfreq, ch_types='eeg')
#         info.set_montage(montage)

#         fig, axes = plt.subplots(1, 2, figsize=(15, 7))
#         plot_topomap(mean_power_original, info, axes=axes[0], show=False)
#         axes[0].set_title("Original Interpolated Data")
#         plot_topomap(mean_power_processed, info, axes=axes[1], show=False)
#         axes[1].set_title("Processed Data")
#         plt.show()



class EDFDataset(Dataset):
    """Integrate multiple EDF files into a dataset."""

    def __init__(self, file_paths, save_path=None, window_size=2.0, step_size=2.0, threshold_std=5, mask_percentage=0.1, l_freq=0.5, h_freq=40.0, rsfreq=256):
        """
        Initialize the EDFDataset class.

        Parameters:
        - file_paths (list of str): List of paths to EDF files.
         - save_path (str): Path to save preprocessed data in HDF5 format.
        - window_size (float): Window size in seconds.
        - step_size (float): Step size between windows in seconds.
        - threshold_std (float): Threshold for clipping based on standard deviation.
        - mask_percentage (float): Percentage of time to mask in the windows.  
        - l_freq (float): Low cutoff frequency for filtering.
        - h_freq (float): High cutoff frequency for filtering.
        - rsfreq (float): Resampling frequency.
        """

        ##########################
        self.datasets, self.dataset_lengths, self.total_length = [], [], 0
        for file_path in file_paths:
            single_dataset = SingleEDFDataset(file_path, window_size, step_size, 
                                              threshold_std, mask_percentage,
                                              l_freq, h_freq, rsfreq)
            self.datasets.append(single_dataset)
            self.dataset_lengths.append(len(single_dataset))

            if save_path:
                single_dataset.save_to_h5(Path(save_path) / f"{Path(file_path).stem}.h5")

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
    
    
    
    