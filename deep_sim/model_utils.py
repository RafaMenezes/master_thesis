import os
import json
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data

def build_mlp(
    input_size,
    layer_sizes,
    output_size=None,
    output_activation=nn.Identity,
    activation=nn.ReLU,
):
    sizes = [input_size] + layer_sizes
    if output_size:
        sizes.append(output_size)

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


def time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]


def get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""
    velocity_sequence = time_diff(position_sequence)
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = torch.randn(list(velocity_sequence.shape)) * (noise_std_last_step/num_velocities**0.5)

    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    position_sequence_noise = torch.cat([
        torch.zeros_like(velocity_sequence_noise[:, 0:1]),
        torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)

    return position_sequence_noise

def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

def compute_connectivity(node_features, n_particles_per_example, radius, device, add_self_edges=True):
        # handle batches. Default is 2 examples per batch

        # Specify examples id for particles/points
        batch_ids =  None#torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(n_particles_per_example)]).to(device)

        # radius = radius + 0.00001 # radius_graph takes r < radius not r <= radius
        edge_index = radius_graph(node_features, r=radius, batch=batch_ids, loop=add_self_edges, max_num_neighbors=500) # (2, n_edges)
        receivers = edge_index[0, :]
        senders = edge_index[1, :]

        return receivers, senders

def sort_edge_index(input_graph):

    edge_index = input_graph.edge_index
    edge_features = input_graph.edge_attr
    # Extract source and target nodes
    source, target = edge_index[0], edge_index[1]

    # Identify self-edges
    self_edges_mask = source == target
    self_edges = edge_index[:, self_edges_mask]
    self_edges_features = edge_features[self_edges_mask]

    # Identify edges where source < target
    normal_edges_mask = source < target
    normal_edges = edge_index[:, normal_edges_mask]
    normal_edges_features = edge_features[normal_edges_mask]

    # Identify edges where source > target
    reverse_edges_mask = source > target
    reverse_edges = edge_index[:, reverse_edges_mask]
    reverse_edges_features = edge_features[reverse_edges_mask]

    # Sort normal edges by source, then by target
    normal_sorted_idx = normal_edges[0] * edge_index.size(1) + normal_edges[1]
    normal_sorted_order = normal_sorted_idx.argsort()
    normal_edges_sorted = normal_edges[:, normal_sorted_order]
    normal_edges_features_sorted = normal_edges_features[normal_sorted_order]

    # Sort reverse edges by target, then by source
    reverse_sorted_idx = reverse_edges[1] * edge_index.size(1) + reverse_edges[0]
    reverse_sorted_order = reverse_sorted_idx.argsort()
    reverse_edges_sorted = reverse_edges[:, reverse_sorted_order]
    reverse_edges_features_sorted = reverse_edges_features[reverse_sorted_order]

    # Concatenate all sorted edges and corresponding features
    sorted_edge_index = torch.cat([self_edges, normal_edges_sorted, reverse_edges_sorted], dim=1)
    sorted_edge_features = torch.cat([self_edges_features, normal_edges_features_sorted, reverse_edges_features_sorted], dim=0)

    # Determine slices
    num_self_edges = self_edges.size(1)
    num_normal_edges = normal_edges_sorted.size(1)
    num_reverse_edges = reverse_edges_sorted.size(1)

    self_edges_slice = (0, num_self_edges)
    normal_edges_slice = (num_self_edges, num_self_edges + num_normal_edges)
    reverse_edges_slice = (num_self_edges + num_normal_edges, num_self_edges + num_normal_edges + num_reverse_edges)

    sorted_graph = Data(
        x = input_graph.x,
        edge_index = sorted_edge_index,
        edge_attr = sorted_edge_features
    )

    return sorted_graph, self_edges_slice, normal_edges_slice, reverse_edges_slice
