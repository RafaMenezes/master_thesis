import os
import json
import pickle
import torch
from tqdm import tqdm

# from tf_data_reader import prepare_data_from_tfds

INPUT_SEQUENCE_LENGTH = 6


def infer(ds, simulator, model_name, metadata, device='cuda'):
    # with open(data_path+'metadata.json', 'rt') as f:
    #     metadata = json.loads(f.read())

    # num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH

    # ds = prepare_data_from_tfds(data_path=data_path, is_rollout=True)
    eval_rollout_splishsplash_data(ds, simulator, model_name, metadata=metadata, device=device)


def eval_single_rollout(simulator, features, num_steps, device):
    initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]
    ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:]
    
    current_positions = initial_positions
    predictions = []
    for step in range(num_steps):
        next_position = simulator.predict_positions(
            current_positions,
            n_particles_per_example=features['n_particles_per_example'],
            particle_types=features['particle_type'],
        ) # (n_nodes, 2)
        # Update kinematic particles from prescribed trajectory.
        kinematic_mask = (features['particle_type'] == 3).clone().detach().to(device)
        next_position_ground_truth = ground_truth_positions[:, step]
        kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, 2)
        next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)
        predictions.append(next_position)
        current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)
    predictions = torch.stack(predictions) # (time, n_nodes, 2)
    ground_truth_positions = ground_truth_positions.permute(1,0,2)
    loss = (predictions - ground_truth_positions) ** 2
    output_dict = {
        'initial_positions': initial_positions.permute(1,0,2).cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
        'particle_types': features['particle_type'].cpu().numpy(),
    }
    return output_dict, loss


def eval_rollout(ds, simulator, num_steps, num_eval_steps=1, save_results=False, metadata=None, device='cuda'):
    eval_loss = []
    i = 0
    simulator.eval()
    with torch.no_grad():
        for example_i, (features, labels) in enumerate(ds):
            features['position'] = torch.tensor(features['position']).to(device) # (n_nodes, 600, 2)
            features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device)
            features['particle_type'] = torch.tensor(features['particle_type']).to(device)
            labels = torch.tensor(labels).to(device)
            example_rollout, loss = eval_single_rollout(simulator, features, num_steps, device)
            example_rollout['metadata'] = metadata
            eval_loss.append(loss)
            if save_results: 
                example_rollout['metadata'] = metadata
                filename = f'rollout_{example_i}.pkl'
                filename = os.path.join('rollouts/', filename)
                with open(filename, 'wb') as f:
                    pickle.dump(example_rollout, f)
            i += 1
            print(f'{i} / {num_eval_steps}')
            if i >= num_eval_steps:
                break
    simulator.train()
    return torch.stack(eval_loss).mean(0)


def eval_rollout_splishsplash_data(ds, simulator, model_name, metadata=None, device='cuda'):
    initial_positions = ds[0][0]
    current_positions = initial_positions
    ground_truth_rollout = []
    predictions = []

    simulator.eval()

    with torch.no_grad():
        for _, labels in tqdm(ds):
            current_positions = current_positions.to(device)
            labels = labels.to(device)
            # particle types are not really being used, only if we had obstacles
            particle_types = torch.full((current_positions.shape[0],), 5).to(device)
            
            next_position = simulator.predict_positions(
                current_positions,
                None,
                particle_types
            )

            ground_truth_rollout.append(labels)
            predictions.append(next_position)
            current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)

        ground_truth_rollout = torch.stack(ground_truth_rollout)
        predictions = torch.stack(predictions)

        loss = (predictions - ground_truth_rollout) ** 2

        example_rollout = {
            'initial_positions': initial_positions.permute(1,0,2).cpu().numpy(),
            'predicted_rollout': predictions.detach().cpu().numpy(),
            'ground_truth_rollout': ground_truth_rollout.cpu().numpy(),
            'particle_types': particle_types.cpu().numpy(),
        }

        example_rollout['metadata'] = metadata
        filename = f'rollout_{model_name}.pkl'
        filename = os.path.join('rollouts/', filename)
        with open(filename, 'wb') as f:
            pickle.dump(example_rollout, f)