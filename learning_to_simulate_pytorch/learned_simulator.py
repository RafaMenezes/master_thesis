import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, radius_graph
from model_utils import build_mlp, time_diff, sort_edge_index

STRATEGY = 'negate' # chose from ['negate', 'avg', 'avg-negate']

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
        super().__init__()
        self.node_fn = nn.Sequential(*[build_mlp(node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out), 
            nn.LayerNorm(node_out)])
        self.edge_fn = nn.Sequential(*[build_mlp(edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out), 
            nn.LayerNorm(edge_out)])

    def forward(self, input_graph): # global_features
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        return Data(
            x = self.node_fn(input_graph.x),
            edge_index = input_graph.edge_index,
            edge_attr = self.edge_fn(input_graph.edge_attr)
        )

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
        super().__init__(aggr='add')
        self.node_fn = nn.Sequential(*[build_mlp(node_in+edge_out, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out), 
            nn.LayerNorm(node_out)])
        self.edge_fn = nn.Sequential(*[build_mlp(node_in+node_in+edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out), 
            nn.LayerNorm(edge_out)])

    def forward(self, latent_graph):
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        x_residual = latent_graph.x
        e_features_residual = latent_graph.edge_attr

        x, e_features = self.propagate(edge_index=latent_graph.edge_index, x=x_residual, e_features=e_features_residual)

        return Data(
            x = x+x_residual,
            edge_index = latent_graph.edge_index,
            edge_attr = e_features+e_features_residual
        )

    def message(self, edge_index, x_i, x_j, e_features):
        e_features = torch.cat([x_i, x_j, e_features], dim=-1)
        e_features = self.edge_fn(e_features)

        return e_features

    def update(self, x_updated, x, e_features):
        # x_updated: (E, edge_out)
        # x: (E, node_in)
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, e_features
    
class SymmetricInteractionNetwork(MessagePassing):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super().__init__(aggr='max')
        self.node_fn = nn.Sequential(*[build_mlp(node_in+edge_out, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out), 
            nn.LayerNorm(node_out)])
        self.edge_fn = nn.Sequential(*[build_mlp(node_in+node_in+edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out), 
            nn.LayerNorm(edge_out)])

    def forward(self, x, latent_graph, normal_edges_slice, reverse_edges_slice):
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        x_residual = latent_graph.x
        e_features_residual = latent_graph.edge_attr

        x, e_features = self.propagate(edge_index=latent_graph.edge_index, x=x_residual, e_features=e_features_residual, normal_edges_slice=normal_edges_slice, reverse_edges_slice=reverse_edges_slice)

        return Data(
            x = x+x_residual,
            edge_index = latent_graph.edge_index,
            edge_attr = e_features+e_features_residual
        )

    def message(self, edge_index, x_i, x_j, e_features, normal_edges_slice, reverse_edges_slice):
        e_features = torch.cat([x_i, x_j, e_features], dim=-1)
        e_features = self.edge_fn(e_features)

        if STRATEGY == 'avg':
            avg_features = (e_features[normal_edges_slice[0] : normal_edges_slice[1]] + e_features[reverse_edges_slice[0] : reverse_edges_slice[1]])/2

            e_features[normal_edges_slice[0] : normal_edges_slice[1]] = avg_features
            e_features[reverse_edges_slice[0] : reverse_edges_slice[1]] = avg_features

        elif STRATEGY == 'avg-negate':
            avg_features = (e_features[normal_edges_slice[0] : normal_edges_slice[1]] + e_features[reverse_edges_slice[0] : reverse_edges_slice[1]])/2

            e_features[normal_edges_slice[0] : normal_edges_slice[1]] = avg_features
            e_features[reverse_edges_slice[0] : reverse_edges_slice[1]] = -avg_features

        elif STRATEGY == 'negate':
            e_features[reverse_edges_slice[0] : reverse_edges_slice[1]] = -e_features[normal_edges_slice[0] : normal_edges_slice[1]]
        else:
            raise ValueError

        return e_features

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
        super().__init__(aggr='max')
        self.gnn_stacks = nn.ModuleList([
            InteractionNetwork(
                node_in=node_in, 
                node_out=node_out,
                edge_in=edge_in, 
                edge_out=edge_out,
                mlp_num_layers=mlp_num_layers,
                mlp_hidden_dim=mlp_hidden_dim,
            ) for _ in range(num_message_passing_steps)])
        
        # self.symm_layer = SymmetricInteractionNetwork(
        #     node_in=node_in, 
        #     node_out=node_out,
        #     edge_in=edge_in, 
        #     edge_out=edge_out,
        #     mlp_num_layers=mlp_num_layers,
        #     mlp_hidden_dim=mlp_hidden_dim
        # )

    def forward(self, latent_graph_0, normal_edges_slice, reverse_edges_slice):
        latent_graph_prev_k = latent_graph_0
        latent_graph_k = latent_graph_0
        
        for gnn in self.gnn_stacks:
            latent_graph_k = gnn(latent_graph_prev_k)
            latent_graph_prev_k = latent_graph_k

        # Symmetric message passing layer
        # x, e_features = self.symm_layer(x, edge_index, e_features, normal_edges_slice, reverse_edges_slice)

        # The last graph, after `m` steps of message-passing
        latent_graph_m = latent_graph_k
        return latent_graph_m

class Decoder(nn.Module):
    def __init__(
        self, 
        node_in, 
        node_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super().__init__()
        self.node_fn = build_mlp(node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out)

    def forward(self, latent_graph_m):
        # x: (E, node_in)
        return self.node_fn(latent_graph_m.x)

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
        super().__init__()
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

    def forward(self, input_graph, normal_edges_slice, reverse_edges_slice):
        # Encode the input_graph.
        latent_graph_0 = self._encoder(input_graph)

        # Do `m` message passing steps in the latent graphs.
        latent_graph_m = self._processor(latent_graph_0, normal_edges_slice, reverse_edges_slice)

        # Decode from the last latent graph.
        return self._decoder(latent_graph_m)

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
        super().__init__()
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

    def forward(self, position_sequence, n_particles_per_example, particle_types):
        # preprocess (build graph)
        input_graph = self._encoder_preprocessor(position_sequence, n_particles_per_example, particle_types)

        # pass through graph network (encode-process-decode)
        normalized_acceleration = self._encode_process_decode(input_graph)
        
        # postprocess (results from network should be transformed to final positions)
        next_position = self._decoder_postprocessor(normalized_acceleration, position_sequence)
        
        return next_position

    def _encoder_preprocessor(self, position_sequence, n_particles_per_example, particle_types):
        n_total_points = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1] # (n_nodes, 2)
        velocity_sequence = time_diff(position_sequence) # Finite-difference.
        # senders and receivers are integers of shape (E,)
        senders, receivers = self._compute_connectivity(most_recent_position, n_particles_per_example, self._connectivity_radius)
        node_features = []
        # Normalized velocity sequence, merging spatial an time axis.
        velocity_stats = self._normalization_stats['velocity']
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

        normalized_relative_displacements = (
            most_recent_position[senders, :] - most_recent_position[receivers, :]
        ) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = torch.norm(normalized_relative_displacements, dim=-1, keepdim=True)
        edge_features.append(normalized_relative_distances)

        # Data is the efficient graph structure from PyTorch Geometric's data library
        return Data(
            x = torch.cat(node_features, dim=-1),
            edge_index = torch.stack([senders, receivers]),
            edge_attr = torch.cat(edge_features, dim=-1)
        )


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
        acceleration_stats = self._normalization_stats['acceleration']
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
        # edge_index, e_features, _, normal_edges_slice, reverse_edges_slice = sort_edge_index(edge_index, e_features)
        normal_edges_slice = None
        reverse_edges_slice = None
        predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, e_features, normal_edges_slice, reverse_edges_slice)
        next_position = self._decoder_postprocessor(predicted_normalized_acceleration, current_positions)
        return next_position

    def predict_accelerations(self, next_position, position_sequence_noise, position_sequence, n_particles_per_example, particle_types):
        noisy_position_sequence = position_sequence + position_sequence_noise
        node_features, edge_index, e_features = self._build_graph_from_raw(noisy_position_sequence, n_particles_per_example, particle_types)
        # edge_index, e_features, _, normal_edges_slice, reverse_edges_slice = sort_edge_index(edge_index, e_features)
        normal_edges_slice = None
        reverse_edges_slice = None
        predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, e_features, normal_edges_slice, reverse_edges_slice)
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
