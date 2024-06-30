import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, radius_graph
from model_utils import build_mlp, time_diff

class Encoder(nn.Module):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Encoder, self).__init__()
        self.node_fn = nn.Sequential(*[build_mlp(node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out), 
            nn.LayerNorm(node_out)])
        self.edge_fn = nn.Sequential(*[build_mlp(edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out), 
            nn.LayerNorm(edge_out)])

    def forward(self, x, edge_index, e_features): # global_features
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        return self.node_fn(x), self.edge_fn(e_features)

class InteractionNetwork(MessagePassing):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(InteractionNetwork, self).__init__(aggr='add')
        self.node_fn = nn.Sequential(*[build_mlp(node_in+edge_out, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out), 
            nn.LayerNorm(node_out)])
        self.edge_fn = nn.Sequential(*[build_mlp(node_in+node_in+edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out), 
            nn.LayerNorm(edge_out)])

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        x_residual = x
        e_features_residual = e_features
        x, e_features = self.propagate(edge_index=edge_index, x=x, e_features=e_features)
        return x+x_residual, e_features+e_features_residual

    def message(self, edge_index, x_i, x_j, e_features):
        e_features_ij = torch.cat([x_i, x_j, e_features], dim=-1)
        e_features_ij = self.edge_fn(e_features_ij)


        edge_dict = {(i.item(), j.item()): idx for idx, (i, j) in enumerate(zip(edge_index[0], edge_index[1]))}
        reverse_index = [edge_dict[(j.item(), i.item())] for i, j in zip(edge_index[0], edge_index[1])]
        
        # Create a symmetric e_features matrix
        e_features_ji = e_features_ij
        e_features_ji[reverse_index] = -e_features_ij

        return e_features_ji

    def update(self, x_updated, x, e_features):
        # x_updated: (E, edge_out)
        # x: (E, node_in)
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, e_features

class Processor(MessagePassing):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Processor, self).__init__(aggr='max')
        self.gnn_stacks = nn.ModuleList([
            InteractionNetwork(
                node_in=node_in, 
                node_out=node_out,
                edge_in=edge_in, 
                edge_out=edge_out,
                mlp_num_layers=mlp_num_layers,
                mlp_hidden_dim=mlp_hidden_dim,
            ) for _ in range(num_message_passing_steps)])

    def forward(self, x, edge_index, e_features):
        for gnn in self.gnn_stacks:
            x, e_features = gnn(x, edge_index, e_features)
        return x, e_features
        # filtered_edge_index, mask = self.filter_edges(edge_index)
        # filtered_e_features = e_features[mask]

        # for gnn in self.gnn_stacks:
        #     x, filtered_e_features = gnn(x, filtered_edge_index, filtered_e_features)
        
        # e_features[mask] = filtered_e_features
        # # Post-process to handle reverse edges
        # reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        # reverse_edge_map = {tuple(e): idx for idx, e in enumerate(reverse_edge_index.T.tolist())}
        
        # for idx, (i, j) in enumerate(edge_index.T.tolist()):
        #     if idx not in mask:
        #         e_features[reverse_edge_map[(j, i)]] = -e_features[reverse_edge_map[(i, j)]]

        # return x, e_features

    def filter_edges(self, edge_index):
        """
        Filter out half of the edges by keeping only one direction for each pair (i, j).
        Assumes edge_index is of shape (2, E).
        """
        # edge_index_transposed = edge_index.t()

        # # Shuffle the rows
        # shuffled_indices = torch.randperm(edge_index_transposed.size(0))
        # shuffled_edge_index_transposed = edge_index_transposed[shuffled_indices]

        # # Transpose back to original form
        # shuffled_edge_index = shuffled_edge_index_transposed.t()
        # edge_set = set()
        # keep_indices = []

        # for idx, (i, j) in enumerate(zip(shuffled_edge_index[0], shuffled_edge_index[1])):
        #     if (j.item(), i.item()) not in edge_set:
        #         edge_set.add((i.item(), j.item()))
        #         keep_indices.append(idx)

        # mask = torch.tensor(keep_indices, dtype=torch.long)
        # return shuffled_edge_index[:, keep_indices], mask
        edge_set = set()
        keep_indices = []

        for idx, (i, j) in enumerate(zip(edge_index[0], edge_index[1])):
            if (j.item(), i.item()) not in edge_set:
                edge_set.add((i.item(), j.item()))
                keep_indices.append(idx)

        mask = torch.tensor(keep_indices, dtype=torch.long)
        return edge_index[:, keep_indices], mask

class Decoder(nn.Module):
    def __init__(
        self, 
        node_in, 
        node_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Decoder, self).__init__()
        self.node_fn = build_mlp(node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out)

    def forward(self, x):
        # x: (E, node_in)
        return self.node_fn(x)

class EncodeProcessDecode(nn.Module):
    def __init__(
        self, 
        node_in,
        node_out,
        edge_in,
        latent_dim,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(EncodeProcessDecode, self).__init__()
        self._encoder = Encoder(
            node_in=node_in, 
            node_out=latent_dim,
            edge_in=edge_in, 
            edge_out=latent_dim,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._processor = Processor(
            node_in=latent_dim, 
            node_out=latent_dim,
            edge_in=latent_dim, 
            edge_out=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._decoder = Decoder(
            node_in=latent_dim,
            node_out=node_out,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        x, e_features = self._encoder(x, edge_index, e_features)
        x, e_features = self._processor(x, edge_index, e_features)
        x = self._decoder(x)
        return x

class Simulator(nn.Module):
    def __init__(
        self,
        particle_dimension,
        node_in,
        edge_in,
        latent_dim,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
        connectivity_radius,
        boundaries,
        normalization_stats,
        num_particle_types,
        particle_type_embedding_size,
        device='cuda',
    ):
        super(Simulator, self).__init__()
        self._boundaries = boundaries
        self._connectivity_radius = connectivity_radius
        self._normalization_stats = normalization_stats
        self._num_particle_types = num_particle_types

        self._particle_type_embedding = nn.Embedding(num_particle_types, particle_type_embedding_size) # (9, 16)

        self._encode_process_decode = EncodeProcessDecode(
            node_in=node_in,
            node_out=particle_dimension,
            edge_in=edge_in,
            latent_dim=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        self._device = device

    def forward(self):
        pass

    def _build_graph_from_raw(self, position_sequence, n_particles_per_example, particle_types):
        n_total_points = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1] # (n_nodes, 2)
        velocity_sequence = time_diff(position_sequence)
        # senders and receivers are integers of shape (E,)
        senders, receivers = self._compute_connectivity(most_recent_position, n_particles_per_example, self._connectivity_radius)
        node_features = []
        # Normalized velocity sequence, merging spatial an time axis.
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
        flat_velocity_sequence = normalized_velocity_sequence.view(n_total_points, -1)
        node_features.append(flat_velocity_sequence)

        # Normalized clipped distances to lower and upper boundaries.
        # boundaries are an array of shape [num_dimensions, 2], where the second
        # axis, provides the lower/upper boundaries.
        boundaries = torch.tensor(self._boundaries, requires_grad=False).float().to(self._device)
        distance_to_lower_boundary = (most_recent_position - boundaries[:, 0][None])
        distance_to_upper_boundary = (boundaries[:, 1][None] - most_recent_position)
        distance_to_boundaries = torch.cat([distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
        normalized_clipped_distance_to_boundaries = torch.clamp(distance_to_boundaries / self._connectivity_radius, -1., 1.)
        node_features.append(normalized_clipped_distance_to_boundaries)

        if self._num_particle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            node_features.append(particle_type_embeddings)

        # Collect edge features.
        edge_features = []

        # Relative displacement and distances normalized to radius
        # (E, 2)
        # normalized_relative_displacements = (
        #     torch.gather(most_recent_position, 0, senders) - torch.gather(most_recent_position, 0, receivers)
        # ) / self._connectivity_radius
        normalized_relative_displacements = (
            most_recent_position[senders, :] - most_recent_position[receivers, :]
        ) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = torch.norm(normalized_relative_displacements, dim=-1, keepdim=True)
        edge_features.append(normalized_relative_distances)

        return torch.cat(node_features, dim=-1), torch.stack([senders, receivers]), torch.cat(edge_features, dim=-1)

    def _compute_connectivity(self, node_features, n_particles_per_example, radius, add_self_edges=True):
        # handle batches. Default is 2 examples per batch

        # Specify examples id for particles/points
        batch_ids = torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(n_particles_per_example)]).to(self._device)
        # radius = radius + 0.00001 # radius_graph takes r < radius not r <= radius
        edge_index = radius_graph(node_features, r=radius, batch=batch_ids, loop=add_self_edges) # (2, n_edges)
        receivers = edge_index[0, :]
        senders = edge_index[1, :]
        return receivers, senders

    def _decoder_postprocessor(self, normalized_acceleration, position_sequence):
        # The model produces the output in normalized space so we apply inverse
        # normalization.
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = (
            normalized_acceleration * acceleration_stats['std']
        ) + acceleration_stats['mean']

        # Use an Euler integrator to go from acceleration to position, assuming
        # a dt=1 corresponding to the size of the finite difference.
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        new_position = most_recent_position + new_velocity  # * dt = 1
        return new_position

    def predict_positions(self, current_positions, n_particles_per_example, particle_types):
        node_features, edge_index, e_features = self._build_graph_from_raw(current_positions, n_particles_per_example, particle_types)
        predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, e_features)
        next_position = self._decoder_postprocessor(predicted_normalized_acceleration, current_positions)
        return next_position

    def predict_accelerations(self, next_position, position_sequence_noise, position_sequence, n_particles_per_example, particle_types):
        noisy_position_sequence = position_sequence + position_sequence_noise
        node_features, edge_index, e_features = self._build_graph_from_raw(noisy_position_sequence, n_particles_per_example, particle_types)
        predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, e_features)
        next_position_adjusted = next_position + position_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(next_position_adjusted, noisy_position_sequence)
        return predicted_normalized_acceleration, target_normalized_acceleration

    def _inverse_decoder_postprocessor(self, next_position, position_sequence):
        """Inverse of `_decoder_postprocessor`."""
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (acceleration - acceleration_stats['mean']) / acceleration_stats['std']
        return normalized_acceleration

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
