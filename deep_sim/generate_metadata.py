from data_reader import SimulationDataset
from learned_simulator import Simulator
import json
import torch
import lightning as L
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

INPUT_SEQUENCE_LENGTH = 6
DEVICE = 'cpu'
MODE = 'train'
STRATEGY = 'baseline'
noise_std = 6.7e-4

ds = SimulationDataset(device=DEVICE, mode='train', window_size=6)


metadata = {}
# # Assuming 2 dimensions (x and y)
# total_vel_sum = torch.zeros(2)  
# total_acc_sum = torch.zeros(2)  
# total_vel_count = 0
# total_acc_count = 0

# # For standard deviation calculations
# vel_list_x = []
# vel_list_y = []
# acc_list_x = []
# acc_list_y = []

# samples = 0
# # Iterate over the dataset
# for features, labels in tqdm(ds):
#     samples += 1
#     vel_sequence = features[:, 1:] - features[:, :-1]
#     acc_sequence = vel_sequence[:, 1:] - vel_sequence[:, :-1]
    
#     # Calculate means
#     total_vel_sum += torch.sum(vel_sequence, dim=(0, 1))
#     total_acc_sum += torch.sum(acc_sequence, dim=(0, 1))
#     total_vel_count += vel_sequence.numel()  # Only count the values in the velocity sequence
#     total_acc_count += acc_sequence.numel()
    
#     # Store velocity and acceleration sequences for standard deviation calculation
#     vel_list_x.append(vel_sequence[:, 0].reshape(-1))  # Flatten x dimension to 1D tensor for std calculation
#     vel_list_y.append(vel_sequence[:, 1].reshape(-1))  # Flatten y dimension to 1D tensor for std calculation
#     acc_list_x.append(acc_sequence[:, 0].reshape(-1))  # Flatten x dimension to 1D tensor for std calculation
#     acc_list_y.append(acc_sequence[:, 1].reshape(-1))  # Flatten y dimension to 1D tensor for std calculation

# # Compute means
# mean_velocity = np.float64(total_vel_sum / total_vel_count)
# mean_acceleration = np.float64(total_acc_sum / total_acc_count)

# # Calculate standard deviations
# vel_tensor = torch.stack([torch.cat(vel_list_x), torch.cat(vel_list_y)])
# acc_tensor = torch.stack([torch.cat(acc_list_x), torch.cat(acc_list_y)])


# std_velocity = np.float64(torch.std(vel_tensor, dim=1, unbiased=False))
# std_acceleration = np.float64(torch.std(acc_tensor, dim=1, unbiased=False))

# metadata['vel_mean'] = list(mean_velocity)
# metadata['acc_mean'] = list(mean_acceleration)
# metadata['vel_std'] = list(std_velocity)
# metadata['acc_std'] = list(std_acceleration)
# metadata['bounds'] = [[-2.0, 2.0], [0.0, 4.0]]
# metadata['sequence_length'] = len(ds)
# metadata['default_connectivity_radius'] = 0.12
# metadata['dim'] = 2
# metadata['dt'] = 0.0025
# metadata['total_samples'] = samples


initial_vel = torch.zeros(2)
initial_vel_var = torch.zeros(2)
initial_acc = torch.zeros(2)
initial_acc_var = torch.zeros(2)

for features, labels in ds:
    vel_sequence = features[:, 1:] - features[:, :-1]
    acc_sequence = vel_sequence[:, 1:] - vel_sequence[:, :-1]
    initial_vel = torch.mean(vel_sequence[:, :4], dim=(0,1))
    initial_acc = torch.mean(acc_sequence[:, :3], dim=(0,1))
    break

k = 0
prior_vel = initial_vel
prior_acc = initial_acc
for features, labels in tqdm(ds):
    k += 1
    vel_sequence = (features[:, 1:] - features[:, :-1])
    most_recent_vel = vel_sequence[:, -1]
    most_recent_vel = torch.mean(most_recent_vel, dim=0)
    vel_correction = (most_recent_vel - prior_vel)
    prior_vel += vel_correction/k
    initial_vel_var += vel_correction * (most_recent_vel - prior_vel)

    most_recent_acc = (vel_sequence[:, 1:] - vel_sequence[:, :-1])[:, -1]
    most_recent_acc = torch.mean(most_recent_acc, dim=0)
    acc_correction = (most_recent_acc - prior_acc)
    prior_acc += acc_correction/k
    initial_acc_var += acc_correction * (most_recent_acc - prior_acc)


metadata['vel_mean'] = list(np.float64(prior_vel))
metadata['acc_mean'] = list(np.float64(prior_acc))
metadata['vel_std'] = list(np.float64(np.sqrt(initial_vel_var/k)))
metadata['acc_std'] = list(np.float64(np.sqrt(initial_acc_var/k)))
metadata['bounds'] = [[0.0, 4.0], [0.0, 4.0]]
metadata['sequence_length'] = len(ds)
metadata['default_connectivity_radius'] = 0.05
metadata['dim'] = 2
metadata['dt'] = 0.0025
metadata['total_samples'] = k




with open('new_metadata.json', 'w') as f:
    json.dump(metadata, f)