import os
import meshio
import torch
import numpy as np
from torch_geometric.data import Data, Dataset


class SimulationDataset(Dataset):
    def __init__(
            self, 
            device='cuda',
            mode='train', 
            window_size=6
        ):
        super().__init__()
        self.device = device
        self.window_size = window_size
        self.mode = mode

        self.data_windows, self.labels, self.n_particles_per_example = self.load_simulation_files(os.path.join('data', mode))

    def extract_number(self, filename):
    # Extract the numerical part of the filename
        return int(filename.split('_')[-1].split('.')[0])


    def load_simulation_files(self, data_dir):
        data_windows = []
        labels = []
        n_particles_per_example = []

        for sim_folder in os.listdir(data_dir):
            sub_data_dir = os.path.join(data_dir, sim_folder,'vtk')

            mesh_files = sorted([f for f in os.listdir(sub_data_dir) if f.endswith('.vtk')], key=self.extract_number)
            all_positions = []
            for mesh_file in mesh_files:
                mesh = meshio.read(os.path.join(sub_data_dir, mesh_file))
                xy_coordinates = mesh.points[:, :2].astype('float32')
                
                positions = torch.tensor(xy_coordinates, dtype=torch.float)
                all_positions.append(positions)

            sim_data_windows, sim_labels, sim_n_particles_per_example = self.create_sliding_windows(all_positions)
            data_windows.extend(sim_data_windows)
            labels.extend(sim_labels)
            n_particles_per_example.extend(sim_n_particles_per_example)
        
        return data_windows, labels, n_particles_per_example

    def create_sliding_windows(self, positions_list):
        data_windows = []
        labels = []
        n_particles_per_example = []
        num_frames = len(positions_list)
        for i in range(num_frames - self.window_size):
            window = positions_list[i:i+self.window_size]
            label = positions_list[i+self.window_size]
            data_windows.append(torch.stack(window).to(self.device))
            labels.append(label.to(self.device))
            # Get number of particles (first dimension of positions tensor)
            num_particles = len(window[0])
            n_particles_per_example.append(torch.tensor([num_particles]).to(self.device))
        return data_windows, labels, n_particles_per_example


    def __len__(self):
        return len(self.data_windows)
    
    def __getitem__(self, idx):
        data_window = self.data_windows[idx].permute(1,0,2)
        label = self.labels[idx]
        n_particles = self.n_particles_per_example[idx]
        return data_window, label #, n_particles
