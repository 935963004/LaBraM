import os
import random
import torch
from torch.utils.data import Dataset
from torch.nn import TripletMarginLoss

class TUABTripletDataset(Dataset):
    def __init__(self, normal_path, abnormal_path, transform=None):
        self.normal_files = [os.path.join(normal_path, f) for f in os.listdir(normal_path) if f.endswith('.pt')]
        self.abnormal_files = [os.path.join(abnormal_path, f) for f in os.listdir(abnormal_path) if f.endswith('.pt')]
        self.transform = transform

    def __len__(self):
        return min(len(self.normal_files), len(self.abnormal_files))

    def __getitem__(self, idx):
        if random.random() < 0.5:
            anchor_files = self.normal_files
            positive_files = self.normal_files
            negative_files = self.abnormal_files
        else:
            anchor_files = self.abnormal_files
            positive_files = self.abnormal_files
            negative_files = self.normal_files

        anchor_file = random.choice(anchor_files)
        positive_file = random.choice(positive_files)
        negative_file = random.choice(negative_files)

        anchor = torch.load(anchor_file)
        positive = torch.load(positive_file)
        negative = torch.load(negative_file)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative