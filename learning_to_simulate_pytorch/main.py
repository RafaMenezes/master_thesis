import os
import json
import numpy as np
import torch
import argparse

from learned_simulator import Simulator
from train import train
from infer import infer

noise_std = 6.7e-4


def main():
    parser = argparse.ArgumentParser(description="Train or infer based on given model.")
    parser.add_argument("--mode", required=True, choices=["train", "eval"], help="Mode to run the model in.")
    parser.add_argument("--training_steps", default=1e6, help="Number of training steps to utilize in training mode.")
    parser.add_argument("--model_path", default="model.pth", help="In case of training, where to save the model. Else, where to load from.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="CPU or GPU (cuda).")
    args = parser.parse_args()

    os.makedirs('train_log', exist_ok=True)
    os.makedirs('rollouts', exist_ok=True)

    with open('data/metadata.json', 'rt') as f:
        metadata = json.loads(f.read())

    normalization_stats = {
        'acceleration': {
            'mean':torch.FloatTensor(metadata['acc_mean']).to(args.device), 
            'std':torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 + noise_std**2).to(args.device),
        }, 
        'velocity': {
            'mean':torch.FloatTensor(metadata['vel_mean']).to(args.device), 
            'std':torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 + noise_std**2).to(args.device),
        }, 
    }

    simulator = Simulator(
        particle_dimension=2,
        node_in=30,
        edge_in=3,
        latent_dim=128,
        num_message_passing_steps=12,
        mlp_num_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=metadata['default_connectivity_radius'],
        boundaries=np.array(metadata['bounds']),
        normalization_stats=normalization_stats,
        num_particle_types=9,
        particle_type_embedding_size=16,
        device=args.device,
    )

    if args.device == "cuda":
        simulator.cuda()

    if args.mode == "train":
        train(
            simulator, 
            training_steps=int(args.training_steps), 
            data_path='data/train.tfrecord', 
            model_path=args.model_path,
            device=args.device
        )
    else:
        simulator.load(args.model_path)
        infer(
            simulator, 
            data_path='data/valid.tfrecord',
            device=args.device
        )

if __name__ == '__main__':
    main()
