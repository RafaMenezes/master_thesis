import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout_dir', default='rollouts_to_evaluate/')

    args = parser.parse_args()

    all_rollout_data = []
    all_chamfer_dist = []
    all_mse = []

    for rollout in os.listdir(args.rollout_dir):
        with open(os.path.join(args.rollout_dir, rollout), "rb") as file:
            rollout_data = pickle.load(file)
        all_rollout_data.append(rollout_data)

        # Calculate chamfer distance
        chamfer_dist = []
        for i in range(rollout_data['predicted_rollout'].shape[0]):
            if i >= 1000: break
            
            chamfer_dist.append(chamfer_distance(rollout_data['predicted_rollout'][i], rollout_data['ground_truth_rollout'][i]))
        
        all_chamfer_dist.append(chamfer_dist)

        # Calculate MSE
        all_mse.append(np.mean(( rollout_data['predicted_rollout'][:1000, :, :] - rollout_data['ground_truth_rollout'][:1000, :, :] ) ** 2, axis=1).sum(axis=1) )

    # Plot results
    plot_mse(all_mse)
    plot_chamfer_dist(all_chamfer_dist)

def plot_chamfer_dist(chamfer_dist_list):
    cmap = plt.get_cmap('tab10')  # 'tab10' gives up to 10 distinct colors, you can change this
    colors = [cmap(i) for i in range(len(chamfer_dist_list))]
    for i, mse in enumerate(chamfer_dist_list):
        steps = np.arange(len(mse))
        plt.plot(steps, mse, label=f'MSE (model {i+1})', color=colors[i],
                 marker='o', linestyle='-', linewidth=2, markersize=2)

    plt.xlabel('Rollout step', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title('Comparison of mean squared error between basic and last layer negated model', fontsize=16)

    # Place the legend a bit lower
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)  # Adjust the position slightly lower

    plt.grid(True)

    # Adjust the subplot parameters to give more space on the bottom
    plt.subplots_adjust(bottom=0.2)  # Increase bottom margin
    plt.savefig(os.path.join(os.path.abspath(os.path.curdir), 'images/chamfer_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_mse(mse_list):
    cmap = plt.get_cmap('tab10')  # 'tab10' gives up to 10 distinct colors, you can change this
    colors = [cmap(i) for i in range(len(mse_list))]
    for i, mse in enumerate(mse_list):
        steps = np.arange(len(mse))
        plt.plot(steps, mse, label=f'MSE (model {i+1})', color=colors[i],
                 marker='o', linestyle='-', linewidth=2, markersize=2)

    plt.xlabel('Rollout step', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title('Comparison of mean squared error between basic and last layer negated model', fontsize=16)

    # Place the legend a bit lower
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)  # Adjust the position slightly lower

    plt.grid(True)

    # Adjust the subplot parameters to give more space on the bottom
    plt.subplots_adjust(bottom=0.2)  # Increase bottom margin
    plt.savefig(os.path.join(os.path.abspath(os.path.curdir), 'images/mse.png'), dpi=300, bbox_inches='tight')
    plt.close()

def chamfer_distance(pred, gt):
    tree_pred = cKDTree(pred)
    tree_gt = cKDTree(gt)
    
    dist_gt_to_pred, _ = tree_pred.query(gt)
    dist_pred_to_gt, _ = tree_gt.query(pred)
    
    return np.mean(dist_gt_to_pred) + np.mean(dist_pred_to_gt)


if __name__ == '__main__':
    main()
