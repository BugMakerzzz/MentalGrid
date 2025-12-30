import os
import random
import yaml
import numpy as np
import magnum as mn
from PIL import Image
from tqdm import tqdm
from box import Box
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb


class HabitatSimulator:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.sim = None
        self.semantic_dic = self._load_semantic_dic()
        self.existing_points = []
        self.agent = None 
    
    def _load_semantic_dic(self):
        data = load_json_data(self.config.semantic_path)
        semantic_dic = {}
        if config.name in ['ArchitecTHOR', 'iTHOR', 'ProcTHOR', 'RoboTHOR']:
            for k, v in data.items():
                semantic_dic[v] = k
        else:            
            for item in data['classes']:
                semantic_dic[item['id']] = item['name']
        return semantic_dic
    
    def _make_sim_cfg(self, scene_name):
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.create_renderer = True
        cfg.default_agent_id = 0
        cfg.gpu_device_id = 0
        cfg.scene_id = scene_name
        cfg.scene_dataset_config_file = self.args.config_path
        cfg.enable_physics = True
        
        cfg.navmesh_settings = habitat_sim.nav.NavMeshSettings()
        cfg.navmesh_settings.set_defaults()
        cfg.navmesh_settings.agent_radius = 0.1
        cfg.navmesh_settings.agent_height = self.args.sensor_height
        cfg.navmesh_settings.include_static_objects = True
        cfg.random_seed = self.args.seed
        
        return cfg
    
    
    def _make_agent_cfg(self):
        sensor_specs = []
        color_sensor_spec = create_camera_spec(
            uuid="color_sensor",
            hfov=self.args.fov,
            far=1000.0,
            sensor_type=habitat_sim.SensorType.COLOR,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
            clear_color=mn.Color4.from_linear_rgb_int(0),
            height=self.args.height,
            width=self.args.width,
            sensor_height=self.args.sensor_height
        )
        sensor_specs.append(color_sensor_spec)
        
        semantic_sensor_spec = create_camera_spec(
            uuid="semantic_sensor",
            hfov=self.args.fov,
            far=1000.0,
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
            height=self.args.height,
            width=self.args.width,
            sensor_height=self.args.sensor_height,
            channels=1,
            semantic_target = habitat_sim._ext.habitat_sim_bindings.SemanticSensorTarget.OBJECT_ID
        )
        sensor_specs.append(semantic_sensor_spec)
        # --- agent + simulator config ---
        cfg = habitat_sim.agent.AgentConfiguration()
        cfg.height = self.args.sensor_height
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
        cfg.height = self.args.sensor_height
        
        return cfg
    
    def _setup_simulator(self, scene_name):
        if self.sim:
            self.sim.close(destroy=True)
        
        sim_cfg = self._make_sim_cfg(scene_name)
        agent_cfg = self._make_agent_cfg()
        
        # Simulator
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    
    def _is_too_close(self, new_point, min_dist=1.0):
        for p in self.existing_points:
            dist = np.linalg.norm(np.array(new_point) - np.array(p))
            if dist < min_dist:
                return True
        return False
    
    def get_observed_obj(self, obs):
        semantic_obs = obs['semantic_sensor']
        unique_ids, counts = np.unique(semantic_obs, return_counts=True)
        mask = unique_ids > 0
        visible_id_counts = dict(zip(unique_ids[mask], counts[mask]))
        
        rigid_obj_mgr = self.sim.get_rigid_object_manager()
        visible_objs = []
        
        for obj_id, pixel_count in visible_id_counts.items():
            visibility_score = pixel_count / semantic_obs.size

            if visibility_score > 0.005: 
                if rigid_obj_mgr.get_library_has_id(obj_id):
                    obj = rigid_obj_mgr.get_object_by_id(obj_id)
                    visible_objs.append(obj)
                    
        return visible_objs
        
        # handles = self.sim.rigid_obj_mgr.get_object_handles()
        # objects = [rigid_obj_mgr.get_object_by_handle(handle) for handle in handles]   
        # semantic_obs = obs['semantic_sensor']
        # visible_elements = semantic_obs[semantic_obs > 0]
        # visible_ids = np.unique(visible_elements)
        # visible_objs = []
        # for obj in objects:
        #     if obj.object_id in visible_ids:
        #         visible_objs.append(obj)
                
        # return visible_objs
    
    def _is_similar_or_duplicate(self, obj_id, existing_ids):
        ID_TO_GROUP = {obj_id: group for group, ids in self.config.similar_groups.items() for obj_id in ids}
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
    
    
    def _filter_objects(self, objects):
        final_objects = []
        agent_transform = self.sim.agents[0].scene_node.transformation_matrix().inverted_rigid()
        existing_semantic_ids = []
        for obj in objects:
            offset = agent_transform.transform_point(obj.translation)
            
            in_range = (self.config.min_x <= offset.x <= self.config.max_x) and \
                    (0 <= offset.y <= 3.0) and \
                    (self.config.min_z <= offset.z <= self.config.max_z)
            if not in_range:
                continue
                
            semantic_id = obj.semantic_id
            if semantic_id in self.config.exclude_index:
                continue
            if self._is_similar_or_duplicate(semantic_id, existing_semantic_ids):
                continue
            
            existing_semantic_ids.append(semantic_id)
            final_objects.append(obj)
            
        return final_objects
      
    
    def _sample_valid_observation(self):
        for _ in range(100):
            pos = self.sim.pathfinder.get_random_navigable_point()
            
            # 1. 基础合法性过滤
            if pos[1] > 0.5: 
                continue
            if self._is_too_close(pos, min_dist=1.0): 
                continue

            # 2. 设置状态并观察
            agent = self.sim.get_agent(0)
            state = agent.get_state()
            state.position = pos
            state.rotation = [0.0, 0.0, 0.0, 1.0]
            agent.set_state(state)
            
            obs = self.sim.get_sensor_observations()
            # 3. 可见度初筛 + 空间/语义精筛
            visible_objs = self._get_visible_objects(obs)
            filtered_objs = self._filter_objects(visible_objs)

            if len(filtered_objs) >= self.config.min_obj_cnt:
                # 采样成功前，清理场景中不可见的物体 (对应你原有的 remove 逻辑)
                # 注意：这会改变仿真环境，通常建议仅在导出前做一次
                return True, obs, filtered_objs
        
        return False, None, None
    
    def run_scene(self, scene_name):
        self._setup_simulator(scene_name)
        self.existing_points = []
        for i in tqdm(range(self.config.n_samples), desc=f"Scene: {scene_name}"):
            self.agent = self.sim.initialize_agent(0)
            state = self._sample_agent_position()
            if not state:
                continue
            
            rigid_obj_mgr = self.sim.get_rigid_object_manager()
            obs = self.sim.get_sensor_observations()
            obs_objects = self.get_observed_obj(rigid_obj_mgr, obs)
            objects = process_scenes(config, sim, obs_objects)
            
            for _ in range(100): # 尝试 100 次采样点
                pos = self.sim.pathfinder.get_random_navigable_point()
                if is_point_too_close(pos, self.existing_points): 
                    continue
                
                # 设置 Agent 状态
                agent = self.sim.get_agent(0)
                state = agent.get_state()
                state.position = pos
                agent.set_state(state)
                
                obs = self.sim.get_sensor_observations()
                raw_objs = self.get_visible_objects(obs)
                filtered_objs = self.filter_objects_by_rules(raw_objs)
                
                if len(filtered_objs) >= self.config.min_obj_cnt:
                    self.existing_points.append(pos)
                    self.save_data(i, scene_name, obs, filtered_objs)
                    success = True
                    break
            
    
    
    def get_visible_objects(self, obs):
        """获取当前视野内可见的物体对象列表"""
        semantic_obs = obs['semantic_sensor']
        visible_ids = np.unique(semantic_obs[semantic_obs > 0])
        
        rigid_obj_mgr = self.sim.get_rigid_object_manager()
        visible_objs = []
        for obj_id in visible_ids:
          
            if rigid_obj_mgr.get_library_has_id(obj_id):
                visible_objs.append(rigid_obj_mgr.get_object_by_id(obj_id))
        return visible_objs