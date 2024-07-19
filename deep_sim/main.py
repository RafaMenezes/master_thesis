import os
import json
import numpy as np
import torch
import argparse
import lightning as L

from learned_simulator import Simulator
from data_reader import SimulationDataset, generate_metadata
# from train import train
# from infer import infer


def main():
    parser = argparse.ArgumentParser(description='Train or infer based on given model.')
    parser.add_argument('--mode', required=True, choices=['train', 'eval'], help='Mode to run the model in.')
    parser.add_argument('--training_steps', default=1e6, help='Number of training steps to utilize in training mode.')
    parser.add_argument('--model_path', default='model.pth', help='In case of training, where to save the model. Else, where to load from.')
    parser.add_argument('--strategy', default='baseline', choices=['baseline', 'mc'], help='Strategy to learn (baseline or mc).')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='CPU or GPU (cuda).')
    args = parser.parse_args()

    os.makedirs('train_log', exist_ok=True)
    os.makedirs('rollouts', exist_ok=True)

    with open('new_metadata.json', 'rt') as f:
        metadata = json.loads(f.read())

    ds = SimulationDataset(device=args.device, data_dir='../learning_to_simulate_pytorch/vtk', window_size=6)
    generate_metadata(ds)

    simulator = Simulator(
        particle_dimension=2,
        node_in=30,
        edge_in=3,
        latent_dim=128,
        num_message_passing_steps=10,
        mlp_num_layers=2,
        mlp_hidden_dim=128,
        num_particle_types=9,
        particle_type_embedding_size=16,
        metadata=metadata,
        dataset=ds,
        strategy=args.strategy,
        device=args.device,
    )

    if args.device == 'cuda':
        simulator.cuda()

    trainer = L.Trainer(max_steps=int(args.training_steps), accelerator=args.device)

    if args.mode == 'train':
        trainer.fit(model=simulator, train_dataloaders=ds)
    else:
        # simulator.load(args.model_path)
        # infer(
        #     simulator, 
        #     data_path='data/',
        #     device=args.device
        # )
        print("only train implemented for now")

if __name__ == '__main__':
    main()
