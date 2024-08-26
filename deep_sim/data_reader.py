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
            window_size=6  # 5 past frames + 1 current frame
        ):
        super().__init__()
        self.device = device
        self.window_size = window_size
        self.mode = mode
        self.data_dir = os.path.join('data', mode)

        # Metadata to store file paths and indices
        self.file_metadata = self._prepare_file_metadata()
        self.total_length = len(self.file_metadata)

    def extract_number(self, filename):
        return int(filename.split('_')[-1].split('.')[0])

    def _prepare_file_metadata(self):
        metadata = []
        for sim_folder in os.listdir(self.data_dir):
            sub_data_dir = os.path.join(self.data_dir, sim_folder, 'vtk')
            mesh_files = sorted([f for f in os.listdir(sub_data_dir) if f.endswith('.vtk')], key=self.extract_number)
            
            for i in range(len(mesh_files) - self.window_size):
                # Store necessary info to locate data on disk
                metadata.append((sub_data_dir, mesh_files[i:i + self.window_size + 1]))
        return metadata

    def __getitem__(self, idx):
        sub_data_dir, mesh_files = self.file_metadata[idx]
        all_positions = []
        
        for mesh_file in mesh_files:
            mesh = meshio.read(os.path.join(sub_data_dir, mesh_file))
            xy_coordinates = mesh.points[:, :2].astype('float32')
            positions = torch.tensor(xy_coordinates, dtype=torch.float)
            all_positions.append(positions)
        
        # Stack positions over the time dimension (window_size)
        data_window = torch.stack(all_positions[:-1], dim=1).to(self.device)  # Shape: [num_particles, 6, 2]
        label = all_positions[-1].to(self.device)  # Shape: [num_particles, 2]

        return data_window, label

    def __len__(self):
        return self.total_length

    def load_simulation_files(self):
        for idx in range(self.total_length):
            yield self.__getitem__(idx)

    def __iter__(self):
        return iter(self.load_simulation_files())

