import numpy as np
import magnum as mn
import quaternion as qt
import habitat_sim
from data_process.load_data import load_json_data


MIN_VISIBLE_SCORE = 0.001

def create_camera_spec(**kw_config):
    camera_sensor_spec = habitat_sim.CameraSensorSpec()
    camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    camera_sensor_spec.resolution = mn.Vector2i(
        [kw_config["height"], kw_config["width"]]
    )
    camera_sensor_spec.position = mn.Vector3(0, kw_config["sensor_height"], 0)
    for k in kw_config:
        setattr(camera_sensor_spec, k, kw_config[k])
    return camera_sensor_spec


class HabitatSimulator:
    def __init__(self, config):
        self.config = config
        self.sim = None
        self.agent = None
        self.ref_T = None # Transformation matrix for the unified reference frame
        self.semantic_dic = self._load_semantic_dic()
        self.existing_points = []
        
        # Pre-process groups for fast lookup
        self.id_to_group = {
            obj_id: group 
            for group, ids in self.config.similar_groups.items() 
            for obj_id in ids
        }
         
    def _load_semantic_dic(self):
        data = load_json_data(self.config.semantic_path)
        semantic_dic = {}
        if self.config.name in ['ArchitecTHOR', 'iTHOR', 'ProcTHOR', 'RoboTHOR']:
            for k, v in data.items():
                semantic_dic[v] = k
        else:            
            for item in data['classes']:
                semantic_dic[item['id']] = item['name']
        return semantic_dic
    
    def setup_scene(self, scene_name):
        """Initialize simulator for a specific scene"""
        if self.sim:
            self.sim.close(destroy=True)
        
        sim_cfg = self._make_sim_cfg(scene_name)
        agent_cfg = self._make_agent_cfg()
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        self.agent = self.sim.initialize_agent(0)
    
    def _make_sim_cfg(self, scene_name):
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.create_renderer = True
        cfg.default_agent_id = 0
        cfg.gpu_device_id = 0
        cfg.scene_id = scene_name
        cfg.scene_dataset_config_file = self.config.config_path
        cfg.enable_physics = True
        
        cfg.navmesh_settings = habitat_sim.nav.NavMeshSettings()
        cfg.navmesh_settings.set_defaults()
        cfg.navmesh_settings.agent_radius = 0.1
        cfg.navmesh_settings.agent_height = self.config.sensor_height
        cfg.navmesh_settings.include_static_objects = True
        cfg.random_seed = self.config.seed
        
        return cfg
     
    def _make_agent_cfg(self):
        sensor_specs = []
        color_sensor_spec = create_camera_spec(
            uuid="color_sensor",
            hfov=self.config.fov,
            far=1000.0,
            sensor_type=habitat_sim.SensorType.COLOR,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
            clear_color=mn.Color4.from_linear_rgb_int(0),
            height=self.config.height,
            width=self.config.width,
            sensor_height=self.config.sensor_height
        )
        sensor_specs.append(color_sensor_spec)
        
        semantic_sensor_spec = create_camera_spec(
            uuid="semantic_sensor",
            hfov=self.config.fov,
            far=1000.0,
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
            height=self.config.height,
            width=self.config.width,
            sensor_height=self.config.sensor_height,
            channels=1,
            semantic_target = habitat_sim._ext.habitat_sim_bindings.SemanticSensorTarget.OBJECT_ID
        )
        sensor_specs.append(semantic_sensor_spec)
        # --- agent + simulator config ---
        cfg = habitat_sim.agent.AgentConfiguration()
        cfg.height = self.config.sensor_height
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
        cfg.height = self.config.sensor_height
        
        return cfg
    
    
    def init_reference_frame(self):
        """Sets the current camera pose as the reference origin (0,0,0)."""
        # Capture the inverted transformation matrix of the first frame
        self.ref_T = self.agent.scene_node.transformation_matrix().inverted_rigid()
   
   
    def set_agent_state(self, position, rotation_deg=0):
        """Updates agent position and orientation."""
        state = self.agent.get_state()
        state.position = position
        angle_rad = np.deg2rad(rotation_deg)
        # qt.from_rotation_vector uses [x, y, z] as axis * angle
        quat = qt.from_rotation_vector(angle_rad * np.array([0, 1, 0]))
        state.rotation = quat

        self.agent.set_state(state)
   
    
    def is_too_close(self, new_point, min_dist=1.5):
        for p in self.existing_points:
            dist = np.linalg.norm(np.array(new_point) - np.array(p))
            if dist < min_dist:
                return True
        return False
    
    
    def _get_visible_objects(self, obs):
        semantic_obs = obs['semantic_sensor']
        # 1. Find unique object IDs present in the current frame (ignoring background 0)
        visible_ids = np.unique(semantic_obs[semantic_obs > 0])
        
        rigid_obj_mgr = self.sim.get_rigid_object_manager()
        visible_objs = []

        # Total pixels for visibility score calculation
        total_pixels = semantic_obs.size
        visible_scores = []
        for obj_id in visible_ids:
            # 2. Directly get object by its ID from the manager
            # Note: Ensure the ID matches the manager's internal ID
            obj = rigid_obj_mgr.get_object_by_id(obj_id)
            if obj is not None:    
                # 3. Calculate visibility score (Optional but recommended)
                pixel_count = np.sum(semantic_obs == obj_id)
                score = pixel_count / total_pixels
                
                visible_scores.append(score)
                visible_objs.append(obj)
                
        return visible_objs, visible_scores
    
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
    
    def _filter_objects(self, objects, visible_scores):
        agent_transform = self.agent.scene_node.transformation_matrix().inverted_rigid()
        existing_semantic_ids = []
        final_objects = []
        
        for obj in objects:
            score = visible_scores[objects.index(obj)]
            if score < MIN_VISIBLE_SCORE:
                continue
            
            offset = agent_transform.transform_point(obj.translation)
            in_range = (self.config.min_x <= offset.x <= self.config.max_x) and \
                    (0 <= offset.y <= 3.0) and \
                    (self.config.min_z <= offset.z <= self.config.max_z)
            if not in_range:
                continue
                
            semantic_id = obj.semantic_id
            if semantic_id in self.config.exclude_index or \
                self._is_similar_or_duplicate(semantic_id, existing_semantic_ids):
                continue
            
            existing_semantic_ids.append(semantic_id)
            final_objects.append(obj)
            
        return final_objects
      
    def capture_step(self):
        """
        Captures observation and converts object locations to the reference frame.
        To be called by generate_data.py for each rotation angle.
        """
        obs = self.sim.get_sensor_observations()
        visible_objs, visible_scores = self._get_visible_objects(obs)
        filtered_objs = self._filter_objects(visible_objs, visible_scores)
        
        view_metadata = []
        for obj in filtered_objs:
            # Transform world coordinates to the unified reference frame (First Frame)
            p_ref = self.ref_T.transform_point(obj.translation)
            
            view_metadata.append({
                'id': obj.object_id,
                'name': self.semantic_dic.get(obj.semantic_id, "unknown").lower(),
                'pos': [p_ref.x, p_ref.y, p_ref.z]
            })
            
        return obs, visible_objs, view_metadata
      
