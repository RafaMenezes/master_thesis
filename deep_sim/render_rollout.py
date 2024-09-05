import pickle
import argparse
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

TYPE_TO_COLOR = {
    3: "black",  # Boundary particles.
    0: "green",  # Rigid solids.
    7: "magenta",  # Goop.
    6: "gold",  # Sand.
    5: "blue",  # Water.
}

def main():
    parser = argparse.ArgumentParser(description="Visualize rollout data.")
    parser.add_argument("--base_rollout_path", required=True, help="Path to rollout baseline pickle file")
    parser.add_argument("--new_rollout_path", required=True, help="Path to rollout new pickle file")
    parser.add_argument("--step_stride", type=int, default=3, help="Stride of steps to skip.")
    parser.add_argument("--block_on_show", type=bool, default=True, help="For test purposes.")
    parser.add_argument("--output_gif", help="Path to save the output GIF.")
    args = parser.parse_args()

    with open(args.base_rollout_path, "rb") as file:
        rollout_data = pickle.load(file)
    with open(args.new_rollout_path, "rb") as file:
        new_rollout_data = pickle.load(file)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    plot_info = []
    for ax_i, (label, rollout_field) in enumerate(
            [("Ground truth", "ground_truth_rollout"),
             ("Baseline", "predicted_rollout"),
             ("Enhanced model", "predicted_rollout")]):
        # Append the initial positions to get the full trajectory.
        if ax_i == 2:
            trajectory = np.concatenate([
                new_rollout_data["initial_positions"],
                new_rollout_data[rollout_field]], axis=0)
        else:
            trajectory = np.concatenate([
                rollout_data["initial_positions"],
                rollout_data[rollout_field]], axis=0)
        ax = axes[ax_i]
        ax.set_title(label)
        bounds = rollout_data["metadata"]["bounds"]
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.)
        points = {
            particle_type: ax.plot([], [], "o", ms=2, color=color)[0]
            for particle_type, color in TYPE_TO_COLOR.items()}
        plot_info.append((ax, trajectory, points))

    num_steps = trajectory.shape[0]
    def update(step_i):
        outputs = []
        for _, trajectory, points in plot_info:
            if step_i > 1000: break
            for particle_type, line in points.items():
                mask = rollout_data["particle_types"] == particle_type
                line.set_data(trajectory[step_i, mask, 0],
                              trajectory[step_i, mask, 1])
                outputs.append(line)
        return outputs

    ani = animation.FuncAnimation(
        fig, update,
        frames=np.arange(0, num_steps, args.step_stride), interval=10)

    if args.output_gif:
        ani.save(args.output_gif, writer='pillow')

    plt.show(block=args.block_on_show)

if __name__ == "__main__":
    main()
