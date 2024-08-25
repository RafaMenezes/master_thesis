import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from model_utils import build_mlp, sort_edge_index

MC_STRATEGY = 'negate' # chose from ['negate', 'avg', 'avg-negate']

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


    def forward(self, latent_graph, normal_edges_slice, reverse_edges_slice):
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
        print('shape of x_i: ', x_i.shape)
        print('shape of x_j: ', x_j.shape)
        print('shape of edge_features: ', e_features.shape)
        e_features = torch.cat([x_i, x_j, e_features], dim=-1)
        print('shape of edge_features after cat: ', e_features.shape)
        e_features = self.edge_fn(e_features)
        print('shape of edge_features after NN: ', e_features.shape)

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


    def forward(self, latent_graph, normal_edges_slice, reverse_edges_slice):
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

        if MC_STRATEGY == 'avg':
            avg_features = (e_features[normal_edges_slice[0] : normal_edges_slice[1]] + e_features[reverse_edges_slice[0] : reverse_edges_slice[1]])/2

            e_features[normal_edges_slice[0] : normal_edges_slice[1]] = avg_features
            e_features[reverse_edges_slice[0] : reverse_edges_slice[1]] = avg_features

        elif MC_STRATEGY == 'avg-negate':
            avg_features = (e_features[normal_edges_slice[0] : normal_edges_slice[1]] + e_features[reverse_edges_slice[0] : reverse_edges_slice[1]])/2

            e_features[normal_edges_slice[0] : normal_edges_slice[1]] = avg_features
            e_features[reverse_edges_slice[0] : reverse_edges_slice[1]] = -avg_features

        elif MC_STRATEGY == 'negate':
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
            ) for _ in range(num_message_passing_steps - 1)])
        
        self.symm_layer = SymmetricInteractionNetwork(
            node_in=node_in, 
            node_out=node_out,
            edge_in=edge_in, 
            edge_out=edge_out,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim
        )


    def forward(self, latent_graph_0, normal_edges_slice, reverse_edges_slice):
        latent_graph_prev_k = latent_graph_0
        latent_graph_k = latent_graph_0
        
        for gnn in self.gnn_stacks:
            latent_graph_k = gnn(latent_graph_prev_k, normal_edges_slice, reverse_edges_slice)
            latent_graph_prev_k = latent_graph_k

        # Symmetric message passing layer, after `m` steps of message-passing
        latent_graph_m = self.symm_layer(latent_graph_k, normal_edges_slice, reverse_edges_slice)

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


class MCGraphNet(nn.Module):
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
    