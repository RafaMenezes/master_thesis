import os
import torch
from pathlib import Path

from model_utils import get_random_walk_noise_for_position_sequence
from tf_data_reader import prepare_data_from_tfds

from torch.utils.tensorboard import SummaryWriter


batch_size = 2
noise_std = 6.7e-4
log_steps = 5
save_steps = 500


def train(
        simulator, 
        training_steps=int(1e6), 
        data_path="data/train.tfrecord", 
        model_path="model.pth", 
        device="cuda"
    ):
    i = 0
    while os.path.isdir('train_log/run'+str(i)):
        i += 1
    LOG_DIR = 'train_log/run'+str(i)+'/'
    Path("model").mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(LOG_DIR)

    lr_init = 1e-4
    lr_min = 1e-6
    lr_decay = 0.1
    lr_decay_steps = int(5e6)
    lr_new = lr_init
    optimizer = torch.optim.Adam(simulator.parameters(), lr=lr_init)

    ds = prepare_data_from_tfds(data_path=data_path, batch_size=batch_size)

    step = 0
    try:
        running_loss = 0.0
        for features, labels in ds:
            features['position'] = torch.tensor(features['position']).to(device)
            features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device)
            features['particle_type'] = torch.tensor(features['particle_type']).to(device)
            labels = torch.tensor(labels).to(device)

            sampled_noise = get_random_walk_noise_for_position_sequence(features['position'], noise_std_last_step=noise_std).to(device)
            non_kinematic_mask = (features['particle_type'] != 3).clone().detach().to(device)
            sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

            pred, target = simulator.predict_accelerations(
                next_position=labels, 
                position_sequence_noise=sampled_noise, 
                position_sequence=features['position'], 
                n_particles_per_example=features['n_particles_per_example'], 
                particle_types=features['particle_type'],
            )
            loss = (pred - target) ** 2
            loss = loss.sum(dim=-1)
            num_non_kinematic = non_kinematic_mask.sum()    

            loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
            loss = loss.sum() / num_non_kinematic

            if step % log_steps == 0:
                writer.add_scalar("training_loss", loss, step)
                writer.add_scalar("lr", lr_new, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

            lr_new = lr_init * (lr_decay ** (step/lr_decay_steps))
            for g in optimizer.param_groups:
                g['lr'] = lr_new

            step += 1
            running_loss += loss.item() / step
            if step >= training_steps:
                break

            if step % save_steps == 0:
                print(f'Training step: {step}/{training_steps}. Loss: {loss}.', end="\r",)
                simulator.save(LOG_DIR+model_path)

    except KeyboardInterrupt:
        pass

    simulator.save(LOG_DIR+model_path)
    print("Model reached end of training. Total running loss was: ", running_loss)
