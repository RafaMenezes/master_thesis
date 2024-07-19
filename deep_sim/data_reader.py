import os
import meshio
import torch
import numpy as np
from torch_geometric.data import Data, Dataset

def generate_metadata(dataset):
    # Initialize accumulators for velocity and acceleration sums
    metadata = {}
    # Assuming 2 dimensions (x and y)
    total_vel_sum = torch.zeros(2).to(dataset.device)
    total_acc_sum = torch.zeros(2).to(dataset.device) 
    total_vel_count = 0
    total_acc_count = 0

    # For standard deviation calculations
    vel_list_x = []
    vel_list_y = []
    acc_list_x = []
    acc_list_y = []

    # Iterate over the dataset
    for features, labels in dataset:
        vel_sequence = features[:, 1:] - features[:, :-1]
        acc_sequence = vel_sequence[:, 1:] - vel_sequence[:, :-1]
        
        # Calculate means
        total_vel_sum += torch.sum(vel_sequence, dim=(0, 1))
        total_acc_sum += torch.sum(acc_sequence, dim=(0, 1))
        total_vel_count += vel_sequence.numel()  # Only count the values in the velocity sequence
        total_acc_count += acc_sequence.numel()
        
        # Store velocity and acceleration sequences for standard deviation calculation
        vel_list_x.append(vel_sequence[:, 0].reshape(-1))  # Flatten x dimension to 1D tensor for std calculation
        vel_list_y.append(vel_sequence[:, 1].reshape(-1))  # Flatten y dimension to 1D tensor for std calculation
        acc_list_x.append(acc_sequence[:, 0].reshape(-1))  # Flatten x dimension to 1D tensor for std calculation
        acc_list_y.append(acc_sequence[:, 1].reshape(-1))  # Flatten y dimension to 1D tensor for std calculation

    # Compute means
    mean_velocity = np.float64(total_vel_sum.cpu().numpy() / total_vel_count)
    mean_acceleration = np.float64(total_acc_sum.cpu().numpy() / total_acc_count)

    # Calculate standard deviations
    vel_tensor = torch.stack([torch.cat(vel_list_x), torch.cat(vel_list_y)])
    acc_tensor = torch.stack([torch.cat(acc_list_x), torch.cat(acc_list_y)])


    std_velocity = np.float64(torch.std(vel_tensor, dim=1, unbiased=False).cpu().numpy())
    std_acceleration = np.float64(torch.std(acc_tensor, dim=1, unbiased=False).cpu().numpy())

    metadata['vel_mean'] = list(mean_velocity)
    metadata['acc_mean'] = list(mean_acceleration)
    metadata['vel_std'] = list(std_velocity)
    metadata['acc_std'] = list(std_acceleration)
    metadata['bounds'] = [[-2.0, 2.0], [0.0, 4.0]]
    metadata['sequence_length'] = len(dataset)
    metadata['default_connectivity_radius'] = 0.05
    metadata['dim'] = 2
    metadata['dt'] = 0.0025

    import json
    with open('new_metadata.json', 'w') as f:
        json.dump(metadata, f)


class SimulationDataset(Dataset):
    def __init__(
            self, 
            device='cuda',
            data_dir='vtk', 
            window_size=6
        ):
        super().__init__()
        self.device = device
        self.data_windows, self.labels, self.n_particles_per_example = self.prepare_data(data_dir, window_size)


    def load_mesh_files(self, data_dir):
        mesh_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.vtk')])
        all_positions = []
        for mesh_file in mesh_files:
            mesh = meshio.read(os.path.join(data_dir, mesh_file))
            xy_coordinates = mesh.points[:, :2].astype('float32')
            positions = torch.tensor(xy_coordinates, dtype=torch.float)
            all_positions.append(positions)
        return all_positions

    def create_sliding_windows(self, positions_list, window_size):
        data_windows = []
        labels = []
        n_particles_per_example = []
        num_frames = len(positions_list)
        for i in range(num_frames - window_size):
            window = positions_list[i:i+window_size]
            label = positions_list[i+window_size]
            data_windows.append(torch.stack(window).to(self.device))
            labels.append(label.to(self.device))
            # Get number of particles (first dimension of positions tensor)
            num_particles = len(window[0])
            n_particles_per_example.append(torch.tensor([num_particles]).to(self.device))
        return data_windows, labels, n_particles_per_example

    def prepare_data(self, data_dir, window_size):
        positions_list = self.load_mesh_files(data_dir)
        data_windows, labels, n_particles_per_example = self.create_sliding_windows(positions_list, window_size)
        return data_windows, labels, n_particles_per_example

    def __len__(self):
        return len(self.data_windows)
    
    def __getitem__(self, idx):
        data_window = self.data_windows[idx].permute(1,0,2)
        label = self.labels[idx]
        n_particles = self.n_particles_per_example[idx]
        return data_window, label #, n_particles
    