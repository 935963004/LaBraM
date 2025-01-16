import os
import pickle
import random
from torch.utils.data import Dataset

class TUABTripletDataset(Dataset):
    def __init__(self, normal_path, abnormal_path):
        self.normal_path = normal_path
        self.abnormal_path = abnormal_path
        self.normal_files = [os.path.join(normal_path, f) for f in os.listdir(normal_path) if f.endswith('.pkl')]
        self.abnormal_files = [os.path.join(abnormal_path, f) for f in os.listdir(abnormal_path) if f.endswith('.pkl')]
        self.all_files = self.normal_files + self.abnormal_files

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        anchor_file = self.all_files[idx]
        anchor_data = self.load_data(anchor_file)
        anchor_label = anchor_data['y']

        if anchor_label == 0:
            positive_file = random.choice(self.normal_files)
            negative_file = random.choice(self.abnormal_files)
        else:
            positive_file = random.choice(self.abnormal_files)
            negative_file = random.choice(self.normal_files)

        positive_data = self.load_data(positive_file)
        negative_data = self.load_data(negative_file)

        return anchor_data['X'], positive_data['X'], negative_data['X']

    def load_data(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data