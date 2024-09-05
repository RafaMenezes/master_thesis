import os
import json
import sys
import numpy as np
import torch
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from torch.utils.data import DataLoader, random_split

from learned_simulator import Simulator
from data_reader import SimulationDataset

from infer import eval_rollout_splishsplash_data

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from tf_record_dataset import TFRecordDataset
# from learning_to_simulate_pytorch.infer import infer


def main():
    parser = argparse.ArgumentParser(description='Train or infer based on given model.')
    parser.add_argument('--mode', required=True, choices=['train', 'test'], help='Mode to run the model in.')
    parser.add_argument('--training_steps', default=500000, help='Number of training steps to utilize in training mode.')
    parser.add_argument('--model_name', default='model', help='In case of training, where to save the model. Else, where to load from.')
    parser.add_argument('--strategy', default='baseline', choices=['baseline', 'mc'], help='Strategy to learn (baseline or mc).')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='CPU or GPU (cuda).')
    args = parser.parse_args()

    # os.makedirs('train_log', exist_ok=True)
    os.makedirs('rollouts', exist_ok=True)

    ds = SimulationDataset(device=args.device, mode=args.mode, window_size=6)
    
    with open('new_metadata.json', 'rb') as f:
        metadata = json.load(f)
    
    if args.mode == 'train':

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
            mode=args.mode,
            strategy=args.strategy,
            device=args.device,
        )

        if args.device == 'cuda':
            simulator.cuda()

        class SaveAtMaxStepsCallback(Callback):
            def on_train_end(self, trainer, pl_module):
                if trainer.global_step >= trainer.max_steps:
                    trainer.save_checkpoint(f"checkpoints/{args.model_name}.ckpt")

        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints/', 
            filename=f'{args.model_name}',  
            save_top_k=1,  
            monitor='validation_loss',
            mode='min'
        )

        trainer = L.Trainer(
            max_steps=int(args.training_steps), 
            accelerator=args.device, 
            enable_checkpointing=True, 
            callbacks=[checkpoint_callback, SaveAtMaxStepsCallback()]
        )

        # train_set = TFRecordDataset(data_path='../learning_to_simulate_pytorch/data/', device=args.device)
        train_set_size = int(len(ds) * 0.9)
        valid_set_size = len(ds) - train_set_size
        seed = torch.Generator().manual_seed(42)
        train_set, val_set = random_split(ds, [train_set_size, valid_set_size], generator=seed)
        torch.multiprocessing.set_start_method('spawn')
        train_loader = DataLoader(train_set, batch_size=2, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=4)

        trainer.fit(model=simulator, train_dataloaders=train_loader, val_dataloaders=val_loader)

    elif args.mode == 'test':
        checkpoint_path = f'checkpoints/{args.model_name}.ckpt'
        simulator = Simulator.load_from_checkpoint(
            checkpoint_path,
            map_location=args.device,
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
            mode=args.mode,
            strategy=args.strategy,
            device=args.device,
        )

        if args.device == 'cuda':
            simulator.cuda()
        eval_rollout_splishsplash_data(ds, simulator, args.model_name, metadata, device=args.device)
        
        # infer(
        #     simulator, 
        #     data_path='../learning_to_simulate_pytorch/data/',
        #     device=args.device
        # ) 
        


if __name__ == '__main__':
    main()
