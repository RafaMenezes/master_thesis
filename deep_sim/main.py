import os
import json
import numpy as np
import torch
import argparse
import lightning as L
from torch.utils.data import DataLoader, random_split

from learned_simulator import Simulator
from data_reader import SimulationDataset, generate_metadata


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

    ds = SimulationDataset(device=args.device, data_dir='data/train', window_size=6)
    metadata = generate_metadata(ds, mode=args.mode)

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
        strategy=args.strategy,
        device=args.device,
    )

    if args.device == 'cuda':
        simulator.cuda()

    trainer = L.Trainer(max_steps=int(args.training_steps), accelerator=args.device, enable_checkpointing=True)

    if args.mode == 'train':
        train_set_size = int(len(ds) * 0.8)
        valid_set_size = len(ds) - train_set_size
        seed = torch.Generator().manual_seed(42)
        train_set, val_set = random_split(ds, [train_set_size, valid_set_size], generator=seed)

        train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)

        trainer.fit(model=simulator, train_dataloaders=train_loader, val_dataloaders=val_loader)
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
