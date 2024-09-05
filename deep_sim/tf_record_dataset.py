import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning_to_simulate_pytorch.tf_data_reader import prepare_data_from_tfds

class TFRecordDataset(Dataset):
    def __init__(self, data_path, is_rollout=False, batch_size=2, device='cuda'):
        self.device = device
        self.ds = prepare_data_from_tfds(data_path, is_rollout)
        self.iterator = iter(self.ds)

    def __len__(self):
        # Since the dataset is potentially infinite, we'll use a large number
        return 1000000  # Adjust this as needed

    def __getitem__(self, idx):
        try:
            features, labels = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.ds)
            features, labels = next(self.iterator)

        # Convert TensorFlow tensors to PyTorch tensors
        features = {k: torch.from_numpy(v) for k, v in features.items() if isinstance(v, np.ndarray)}
        labels = torch.from_numpy(labels)

        features['position'] = features['position'].to(self.device)
        features['n_particles_per_example'] = features['n_particles_per_example'].to(self.device)
        features['particle_type'] = features['particle_type'].to(self.device)
        labels = labels.to(self.device)

        return features, labels
    