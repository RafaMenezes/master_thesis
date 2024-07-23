import json
import os
import shutil
import random
import subprocess
import meshio


def generate_random_configuration():
    translation_x = random.uniform(-2.0, 1.0)
    translation_y = random.uniform(0.0, 1.0)
    
    return {
        'denseMode': 0,
        'start': [0.0, 0.0, -1],
        'end': [1.0, 1.0, 1],
        'translation': [translation_x, translation_y, 0.0],
        'scale': [1, 1, 1]
    }


def main():
    num_simulations = 50
    scenes_path = "/Users/rafajm/.pyenv/versions/3.9.2/envs/simulation/bin/data/Scenes"
    file_path = "DamBreakModel_2D.json"
    output_dir_base = "/Users/rafajm/code/rwth/thesis/master_thesis/deep_sim/data/train"

    with open(os.path.join(scenes_path, file_path), 'r') as f:
        original_data = json.load(f)
            
    for i in range(num_simulations):
        sampled_value = None
        
        new_config = generate_random_configuration()

        original_data['FluidBlocks'] = [new_config]
        
        modified_file_path = os.path.join(scenes_path, f"DamBreakModel_2D_{i}.json")
        
        with open(modified_file_path, 'w') as f:
            json.dump(original_data, f, indent=4)
        
        output_dir = os.path.join(output_dir_base, f"simulation_{i}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Dirty fix -- splish splash is randomly generating nan sequences of positions (in both GUI and --no-gui options)
        while sampled_value == None:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            command = f"splash {modified_file_path} --no-gui --output-dir {output_dir}"
            subprocess.run(command, shell=True, check=True)

            vtk_dir = os.path.join(output_dir,'vtk')
            mesh_files = sorted([f for f in os.listdir(vtk_dir) if f.endswith('.vtk')])
            all_positions = []
            for mesh_file in mesh_files:
                mesh = meshio.read(os.path.join(vtk_dir, mesh_file))
                xy_coordinates = mesh.points[:, :2].astype('float32')
                all_positions.append(xy_coordinates)
            sampled_value = all_positions[0][0][0] if (all_positions[0][0][0] >= 0 or all_positions[0][0][0] < 0) else None
            print("the sampled value was: ", sampled_value)


if __name__ == "__main__":
    main()
