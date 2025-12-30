import numpy as np
from PIL import Image
import habitat_sim
import argparse
import random
import numpy as np
import magnum as mn
import os 
from tqdm import tqdm 
from habitat_sim.utils.common import d3_40_colors_rgb
from eval.load_data import load_json_data, save_json_data
import yaml
from box import Box


def create_camera_spec(**kw_args):
    camera_sensor_spec = habitat_sim.CameraSensorSpec()
    camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    camera_sensor_spec.resolution = mn.Vector2i(
        [kw_args["height"], kw_args["width"]]
    )
    camera_sensor_spec.position = mn.Vector3(0, kw_args["sensor_height"], 0)
    for k in kw_args:
        setattr(camera_sensor_spec, k, kw_args[k])
    return camera_sensor_spec


def make_agent_cfg(args):
    sensor_specs = []
    color_sensor_spec = create_camera_spec(
        uuid="color_sensor",
        hfov=args.fov,
        far=1000.0,
        sensor_type=habitat_sim.SensorType.COLOR,
        sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        clear_color=mn.Color4.from_linear_rgb_int(0),
        height=args.height,
        width=args.width,
        sensor_height=args.sensor_height
    )
    sensor_specs.append(color_sensor_spec)
    
    semantic_sensor_spec = create_camera_spec(
        uuid="semantic_sensor",
        hfov=args.fov,
        far=1000.0,
        sensor_type=habitat_sim.SensorType.SEMANTIC,
        sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        height=args.height,
        width=args.width,
        sensor_height=args.sensor_height,
        channels=1,
        semantic_target = habitat_sim._ext.habitat_sim_bindings.SemanticSensorTarget.OBJECT_ID
    )
    sensor_specs.append(semantic_sensor_spec)
    # --- 构造 agent + simulator config ---
    cfg = habitat_sim.agent.AgentConfiguration()
    cfg.height = args.sensor_height
    cfg.sensor_specifications = sensor_specs
    cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    # construct a NavMeshSettings from default agent paramters for SimulatorConfiguration
    cfg.height = args.sensor_height
    
    return cfg


def make_sim_cfg(args):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.create_renderer = True
    cfg.default_agent_id = 0
    cfg.gpu_device_id = 0
    cfg.scene_id = args.scene
    cfg.scene_dataset_config_file = args.config_path
    cfg.enable_physics = True
    cfg.navmesh_settings = habitat_sim.nav.NavMeshSettings()
    cfg.navmesh_settings.set_defaults()
    cfg.navmesh_settings.agent_radius = 0.1
    cfg.navmesh_settings.agent_height = args.sensor_height
    cfg.navmesh_settings.include_static_objects = True
    cfg.random_seed = args.seed
    return cfg

def is_similar(config, obj_id, existing_ids):
    ID_TO_GROUP = {obj_id: group for group, ids in config.similar_groups.items() for obj_id in ids}
    if obj_id in existing_ids:
        return True
    group_a = ID_TO_GROUP.get(obj_id)
    if not group_a:
        return False 
    for id in existing_ids:
        group_b = ID_TO_GROUP.get(id)
        if group_b and group_b == group_a:
            return True
    return False


def process_scenes(config, sim, objects):
    final_objects = []
    agent_transform = sim.agents[0].scene_node.transformation_matrix().inverted_rigid()
    
    existing_semantic_ids = []
    for obj in objects:
        offset = agent_transform.transform_point(obj.translation)
        if offset.z > config.max_z or offset.z < config.min_z or offset.y < 0 \
            or offset.x > config.max_x or offset.x < config.min_x > 3.0:
            continue
        semantic_id = obj.semantic_id
        if is_similar(config, semantic_id, existing_semantic_ids) or semantic_id in config.exclude_index:
            continue
        else:
            existing_semantic_ids.append(semantic_id)
            
        final_objects.append(obj)
        
    return final_objects


def get_observed_obj(rigid_obj_mgr, obs):
    handles = rigid_obj_mgr.get_object_handles()
    objects = [rigid_obj_mgr.get_object_by_handle(handle) for handle in handles]   
    semantic_obs = obs['semantic_sensor']
    visible_elements = semantic_obs[semantic_obs > 0]
    visible_ids = np.unique(visible_elements)
    visible_objs = []
    for obj in objects:
        if obj.object_id in visible_ids:
            visible_objs.append(obj)
            
    return visible_objs


def load_semantic_dic(config):
    data = load_json_data(config.semantic_path)
    semantic_dic = {}
    if config.name in ['ArchitecTHOR', 'iTHOR', 'ProcTHOR', 'RoboTHOR']:
        for k, v in data.items():
            semantic_dic[v] = k
    else:            
        for item in data['classes']:
            semantic_dic[item['id']] = item['name']
    return semantic_dic


def is_too_close(new_point, existing_points, min_dist):
    for p in existing_points:
        dist = np.linalg.norm(np.array(new_point) - np.array(p))
        if dist < min_dist:
            return True
    return False


def load_scenes(config, split):
    scenes = []
    with open(config.split_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    for name in data[split]:
        scenes.append(name)
    return scenes


def main(args):
    config_name = args.dataset
    config_path = f'configs/data/{config_name}.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = Box(yaml.load(f, Loader=yaml.FullLoader))
    args.config_path = config.config_path
    scenes = load_scenes(config=config, split=args.split)
    for scene in scenes:
        args.scene = scene
        sim_cfg = make_sim_cfg(args)
        agent_cfg = make_agent_cfg(args)
        semantic_dic = load_semantic_dic(config=config)
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        existing_points = []
        for scene_id in tqdm(range(1, config.n_samples + 1)):
            agent = sim.initialize_agent(0)
            agent_state = agent.get_state()
            agent_state.rotation = [0.0, 0.0, 0.0, 1.0]
            num_start_tries = 0
            while agent_state.position[1] > 0.5 and num_start_tries < 100:
                agent_state.position = sim.pathfinder.get_random_navigable_point()
                num_start_tries += 1
            agent.set_state(agent_state)
            agent_state = agent.get_state()
            rigid_obj_mgr = sim.get_rigid_object_manager()
            obs = sim.get_sensor_observations()
            obs_objects = get_observed_obj(rigid_obj_mgr, obs)
            objects = process_scenes(config, sim, obs_objects)
            
            num_start_tries = 0
            while len(objects) < config.min_obj_cnt or is_too_close(agent_state.position, existing_points, 1.0):
                agent_state.position = sim.pathfinder.get_random_navigable_point()
                agent.set_state(agent_state)
                agent_state = agent.get_state()
                obs = sim.get_sensor_observations()
                obs_objects = get_observed_obj(rigid_obj_mgr, obs)
                objects = process_scenes(config, sim, obs_objects)
                
                num_start_tries += 1
                if num_start_tries > 100:
                    break 
                
            if num_start_tries > 100:
                continue
            
            existing_points.append(agent_state.position)
            
            for obj in obs_objects:
                if obj not in objects:
                    rigid_obj_mgr.remove_object_by_handle(obj.handle)

            output_img_dir = f"data/source/{args.dataset}/{args.split}/images/"
            output_location_dir = f"data/source/{args.dataset}/{args.split}/"
            if not os.path.exists(output_img_dir):
                os.makedirs(output_img_dir)
            if not os.path.exists(output_location_dir):
                os.makedirs(output_location_dir)
                
            obs = sim.get_sensor_observations()
            color_obs = obs["color_sensor"]
            color_img = Image.fromarray(color_obs, mode="RGBA")
            color_img.save(os.path.join(output_img_dir, f"{args.scene}-{scene_id}.png"))
            
            if args.semantic:
                semantic_obs = obs["semantic_sensor"]
                semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
                semantic_img.putpalette(d3_40_colors_rgb.flatten())
                semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
                semantic_img.save(os.path.join(output_img_dir, f"{args.scene}-{scene_id}_sem.png"))
                
            # ---------- agent ----------
            agent_data = {
                "position": [
                    float(agent_state.position[0]),
                    float(agent_state.position[1]),
                    float(agent_state.position[2])
                ],
                "rotation": [
                    agent_state.rotation.x,
                    agent_state.rotation.y,
                    agent_state.rotation.z,
                    agent_state.rotation.w
                ]
            }

            objects_data = []
            agent_transform = sim.agents[0].scene_node.transformation_matrix().inverted_rigid()
            for obj in objects:
                offset = agent_transform.transform_point(obj.translation)
                # offset = world_T_cam.transform_point(obj.translation)
                msg = {
                    'id': obj.object_id,
                    'name': semantic_dic[obj.semantic_id].lower(), 
                    'location': [offset.x, offset.y, offset.z]
                }
                objects_data.append(msg)
                
            results = {
                "obj_cnt": len(objects),
                "agent": agent_data,
                "objects": objects_data
            }
            save_json_data(path=os.path.join(output_location_dir, f"{args.scene}-{scene_id}_info.json"), data=results)
            
        sim.close(destroy=True)



parser = argparse.ArgumentParser()
# parser.add_argument("--scene", type=str, default="v3_sc0_staging_00")
parser.add_argument("--dataset", type=str, default="hssd")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--sensor_height", type=float, default=1.5)
parser.add_argument("--fov", type=int, default=110)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--semantic", action='store_true')
args = parser.parse_args()
random.seed(args.seed)

if __name__ == '__main__':
    main(args=args)
    