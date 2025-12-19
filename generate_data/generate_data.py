import numpy as np
from PIL import Image
import habitat_sim
import argparse
import random
import numpy as np
import magnum as mn
import json 
import os 
from tqdm import tqdm 
from habitat_sim.utils.common import d3_40_colors_rgb

CONFIG_PATH_DICT = {
    'hssd': '/mnt/userdata/ljc/data/hssd-hab/hssd-hab.scene_dataset_config.json',
    'replica': '/mnt/userdata/ljc/data/replica_cad/replicaCAD.scene_dataset_config.json',
    'ai2thor': '/mnt/userdata/ljc/data/ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json'
}

SEMANTIC_PATH_DICT = {
    'hssd': '/mnt/userdata/ljc/data/hssd-hab/semantics/hssd-hab_semantic_lexicon.json',
    'replica': '/mnt/userdata/ljc/data/replica_cad/configs/ssd/replicaCAD_semantic_lexicon.json',
    'ai2thor': '/mnt/userdata/ljc/data/ai2thor-hab/ai2thor-hab/configs/object_semantic_id_mapping.json'
}

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
    cfg.scene_dataset_config_file = CONFIG_PATH_DICT[args.dataset]
    cfg.enable_physics = True
    cfg.navmesh_settings = habitat_sim.nav.NavMeshSettings()
    cfg.navmesh_settings.set_defaults()
    cfg.navmesh_settings.agent_radius = 0.1
    cfg.navmesh_settings.agent_height = args.sensor_height
    cfg.navmesh_settings.include_static_objects = True
    cfg.random_seed = args.seed
    
    return cfg

def process_scenes(sim, objects):
    final_objects = []
    agent_transform = sim.agents[0].scene_node.transformation_matrix().inverted_rigid()
    
    exist_semantic_ids = []
    for obj in objects:
        offset = agent_transform.transform_point(obj.translation)
        if offset.z >= 0 or offset.y <= 0 or abs(offset.x) > 3:
            continue
        
        semantic_id = obj.semantic_id
        if semantic_id in exist_semantic_ids:
            continue
        
        exist_semantic_ids.append(semantic_id)
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

def load_semantic_dic(name):
    path = SEMANTIC_PATH_DICT[name]
    data = load_json_data(path)
    semantic_dic = {}
    if name == 'ai2thor':
        for k, v in data.items():
            semantic_dic[v] = k
    else:            
        for item in data['classes']:
            semantic_dic[item['id']] = item['name']
    return semantic_dic

def load_json_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
    return data

def save_json_data(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        f.close()
    return 

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="v3_sc0_staging_00")
parser.add_argument("--dataset", type=str, default="replica")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--sensor_height", type=float, default=1.5)
parser.add_argument("--fov", type=int, default=90)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--n_samples", type=int, default=10)
parser.add_argument("--semantic", action='store_true')
parser.add_argument("--obj_density", type=str, default='mid')
args = parser.parse_args()

if __name__ == '__main__':
    random.seed(args.seed)
    sim_cfg = make_sim_cfg(args)
    agent_cfg = make_agent_cfg(args)
    semantic_dic = load_semantic_dic(args.dataset)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    for scene_id in tqdm(range(1, args.n_samples + 1)):
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
        objects = process_scenes(sim, obs_objects)
        
        if args.obj_density == 'low':
            min_obj, max_obj = 2, 4 
        elif args.obj_density == 'mid':
            min_obj, max_obj = 5, 7
        else:
            min_obj, max_obj = 8, 10
        
        num_start_tries = 0
        while len(objects) < min_obj or len(objects) > max_obj:
            agent_state.position = sim.pathfinder.get_random_navigable_point()
            agent.set_state(agent_state)
            agent_state = agent.get_state()
            obs = sim.get_sensor_observations()
            obs_objects = get_observed_obj(rigid_obj_mgr, obs)
            objects = process_scenes(sim, obs_objects)
            num_start_tries += 1
            if num_start_tries > 100:
                break 
        if num_start_tries > 100:
            continue
        
        for obj in obs_objects:
            if obj not in objects:
                rigid_obj_mgr.remove_object_by_handle(obj.handle)

        output_img_dir = f"data/source/{args.dataset}/{args.scene}/{args.obj_density}_obj/images/"
        output_location_dir = f"data/source/{args.dataset}/{args.scene}/{args.obj_density}_obj/"
        if not os.path.exists(output_img_dir):
            os.makedirs(output_img_dir)
        if not os.path.exists(output_location_dir):
            os.makedirs(output_location_dir)
            
        
        obs = sim.get_sensor_observations()
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        color_img.save(os.path.join(output_img_dir, f"scene{scene_id}.png"))
        
        if args.semantic:
            semantic_obs = obs["semantic_sensor"]
            semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
            semantic_img.putpalette(d3_40_colors_rgb.flatten())
            semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
            semantic_img.save(os.path.join(output_img_dir, f"scene{scene_id}_sem.png"))
            
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
            

            # offset = obj.translation
            msg = {
                'id': obj.object_id,
                'name': semantic_dic[obj.semantic_id], 
                'location': [offset.x, offset.y, offset.z]
            }
            # print(msg)
            objects_data.append(msg)
            
    
                    
        bounds = sim.pathfinder.get_bounds()
        min_bounds = agent_transform.transform_point(bounds[0])
        max_bounds = agent_transform.transform_point(bounds[1])
        print(min_bounds)
        print(max_bounds)        
        results = {
            "agent": agent_data,
            "objects": objects_data
        }
        
        # save_json_data(path=os.path.join(output_location_dir, f"scene{scene_id}.json"), data=results)