import torch
import torch.nn as nn
from torch_geometric.data import Data

from model_utils import time_diff, compute_connectivity, sort_edge_index
from graph_network import EncodeProcessDecode
from mc_graph_network import MCGraphNet

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
        strategy='baseline',
        device='cuda',
    ):
        super().__init__()
        self._boundaries = boundaries
        self._connectivity_radius = connectivity_radius
        self._normalization_stats = normalization_stats
        self._num_particle_types = num_particle_types
        self._strategy = strategy

        self._particle_type_embedding = nn.Embedding(num_particle_types, particle_type_embedding_size) # (9, 16)

        if self._strategy == 'baseline':
            self._graph_network = EncodeProcessDecode(
                node_in=node_in,
                node_out=particle_dimension,
                edge_in=edge_in,
                latent_dim=latent_dim,
                num_message_passing_steps=num_message_passing_steps,
                mlp_num_layers=mlp_num_layers,
                mlp_hidden_dim=mlp_hidden_dim,
            )
        elif self._strategy == 'mc':
            self._graph_network = MCGraphNet(
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
        ## NOT ACTUALLY BEING CALLED YET
        # # preprocess (build graph)
        # input_graph = self._encoder_preprocessor(position_sequence, n_particles_per_example, particle_types)

        # # pass through graph network (encode-process-decode)
        # normalized_acceleration = self._graph_network(input_graph)
        
        # # postprocess (results from network should be transformed to final positions)
        # next_position = self._decoder_postprocessor(normalized_acceleration, position_sequence)
        
        # return next_position
        pass


    def _encoder_preprocessor(self, position_sequence, n_particles_per_example, particle_types):
        n_total_points = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1] # (n_nodes, 2)
        velocity_sequence = time_diff(position_sequence) # Finite-difference.
        # senders and receivers are integers of shape (E,)
        senders, receivers = compute_connectivity(most_recent_position, n_particles_per_example, self._connectivity_radius, self._device)

        node_features = []
        # Normalized velocity sequence, merging spatial an time axis.
        velocity_stats = self._normalization_stats['velocity']
        normalized_velocity_sequence = (velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
        flat_velocity_sequence = normalized_velocity_sequence.reshape(n_total_points, -1)
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

        graph = Data(
            x = torch.cat(node_features, dim=-1),
            edge_index = torch.stack([senders, receivers]),
            edge_attr = torch.cat(edge_features, dim=-1)
        )

        self_edges_slice = None
        normal_edges_slice = None
        reverse_edges_slice  = None
        
        if self._strategy == 'mc':
            graph, self_edges_slice, normal_edges_slice, reverse_edges_slice = sort_edge_index(graph)
        # Data is the efficient graph structure from PyTorch Geometric's data library
        return graph, self_edges_slice, normal_edges_slice, reverse_edges_slice 


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
        input_graph, _, normal_edges_slice, reverse_edges_slice = self._encoder_preprocessor(current_positions, n_particles_per_example, particle_types)

        if self._strategy == 'baseline':
            predicted_normalized_acceleration = self._graph_network(input_graph)
        elif self._strategy == 'mc':
            predicted_normalized_acceleration = self._graph_network(input_graph, normal_edges_slice, reverse_edges_slice)

        next_position = self._decoder_postprocessor(predicted_normalized_acceleration, current_positions)
        return next_position


    def get_predicted_and_target_normalized_accelerations(self, next_position, position_sequence_noise, position_sequence, n_particles_per_example, particle_types):
        # Adds noise to the position sequence -- helps stabilize the errors along the rollout in inference
        noisy_position_sequence = position_sequence + position_sequence_noise

        # Perform a forward pass through the graph network
        input_graph, _, normal_edges_slice, reverse_edges_slice  = self._encoder_preprocessor(noisy_position_sequence, n_particles_per_example, particle_types)

        if self._strategy == 'baseline':
            predicted_normalized_acceleration = self._graph_network(input_graph)
        elif self._strategy == 'mc':
            predicted_normalized_acceleration = self._graph_network(input_graph, normal_edges_slice, reverse_edges_slice)

        # By adding position_sequence_noise[:, -1] to the NEXT position (target), the inverse_decoder
        # guarantees that the target acceleration is the difference between a next correct velocity and
        # a previous noisy one -- the model should, thus, learn this correction.
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
