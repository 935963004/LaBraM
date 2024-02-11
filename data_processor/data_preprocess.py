import random
import numpy as np
import torch

def mask_channels(data, channels=[1, 2, 3]):
    # mask = torch.from_numpy(np.zeros((data.shape[0], len(channels), data.shape[2])))
    data[:, channels, :] = 0
    return data

def normalization(data):
    return data / 100

def collate_mask_time(data, mask_percentage):
    data = torch.from_numpy(np.array(data)) / 100
    data_len = data.shape[-1]
    mask_start_idx = random.randint(0, int(data_len * (1-mask_percentage)))
    mask_end_idx = mask_start_idx + int(data_len*mask_percentage)
    masked_data = data.clone()
    masked_data[:, :, mask_start_idx:mask_end_idx] = 0
    return masked_data, data, [mask_start_idx, mask_end_idx]