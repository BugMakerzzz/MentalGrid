import os
import random
from PIL import Image
from tqdm import tqdm
import argparse
from data_process.load_data import save_json_data
import yaml
from box import Box
from space.habitat import HabitatSimulator



def load_scenes(config, split):
    scenes = []
    with open(config.split_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    for name in data[split]:
        scenes.append(name)
        
    if split == 'train':
        return scenes[:1000]
    else:
        return scenes[:100]


def run_scene_generation(simulator, scene, output_dir):
    count = 0
    attempts = 0
    simulator.existing_points = [] # Reset for each scene
    
    pbar = tqdm(total=simulator.config.n_samples, desc=f"Scene: {scene}")

    while count < simulator.config.n_samples:
        if attempts == 0:
            num_views = random.randint(1, 5)
            direction = random.choice([1, -1])
            overlap_factor = random.choice([0.4, 0.5, 0.6])
            
            fov = simulator.config.fov
            step = fov * overlap_factor * direction
            
            simulator.setup_scene(scene)
            
        if attempts > 100:
            count += 1
            attempts = 0
            continue
            
        pos = simulator.sim.pathfinder.get_random_navigable_point()
        if pos[1] > 0.5 or simulator.is_too_close(pos, min_dist=2.0):
            attempts += 1
            continue

        results = []
        existing_names = set()
        all_seen_obj = set()
        keep_obj_ids = set() 
        for v_idx in range(num_views):
            current_angle = v_idx * step
            simulator.set_agent_state(pos, rotation_deg=current_angle)
            if v_idx == 0:
                simulator.init_reference_frame()
                
            obs, seen_obj, metadata = simulator.capture_step()
            all_seen_obj |= set(seen_obj)
        
            # Check if this sample is worth keeping (at least one view sees objects)
            if not metadata and v_idx == 0:
                break
            
            msg = []
            for obj in metadata:
                if obj['name'] not in existing_names or obj['id'] in keep_obj_ids:
                    keep_obj_ids.add(obj['id'])
                    msg.append(obj)
                    existing_names.add(obj['name'])
            
            results.append((msg, current_angle))
        
        #  Save data if validation passes
        if len(keep_obj_ids) >= simulator.config.min_obj_cnt:
            count += 1
            
            sample_img_dir = os.path.join(output_dir, "images", f"{scene}_s{count}")
            os.makedirs(sample_img_dir, exist_ok=True)
            
  
            rigid_obj_mgr = simulator.sim.get_rigid_object_manager()
            for obj in all_seen_obj:
                id = obj.object_id
                handle = obj.handle
                if id not in keep_obj_ids:
                    rigid_obj_mgr.remove_object_by_handle(handle)
    
            agent_info = {
                "position": [
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2])
                ],
                "rotation": []
            }
            
            object_info = {}
            for v_idx, (metadata, angle) in enumerate(results):
                simulator.set_agent_state(pos, rotation_deg=angle)
                obs = simulator.sim.get_sensor_observations()
                
                agent_info["rotation"].append(angle)
                # Save RGB
                Image.fromarray(obs["color_sensor"], mode="RGBA").save(
                    os.path.join(sample_img_dir, f"{v_idx}.png")
                )
                
                for item in metadata:
                    if item['id'] not in object_info:
                        object_info[item['id']] = {
                            'name': item['name'],
                            'pos': item['pos'],
                            'view': [v_idx]
                        }
                    else:
                        object_info[item['id']]['view'].append(v_idx)
                # Save Metadata (All locations are relative to v0 camera)
            info = {
                'view_cnt': len(results),
                'view_step': step,
                'obj_cnt': len(keep_obj_ids),
                'agent': agent_info,
                'object': object_info
            }
            save_json_data(os.path.join(output_dir, f"{scene}_s{count}_info.json"), info)

            simulator.existing_points.append(pos)
            
            attempts = 0
            pbar.update(1)
            
        else:
            attempts += 1


def main(args):
    config_name = args.dataset
    config_path = f'configs/data/{config_name}.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = Box(yaml.load(f, Loader=yaml.FullLoader))
    args.config_path = config.config_path
    scenes = load_scenes(config=config, split=args.split)
    
    simulator = HabitatSimulator(config)
    
    output_dir = f"data/scene/{args.dataset}/{args.split}"
    
    for scene_name in scenes: 
        run_scene_generation(simulator=simulator, scene=scene_name, output_dir=output_dir)



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="hssd")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--seed", type=int, default=17)
args = parser.parse_args()
random.seed(args.seed)

if __name__ == '__main__':
    main(args=args)
    