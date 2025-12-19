import math 
from utils.template import Direction, ALLO_DIRECTIONS, EGO_DIRECTIONS, Movement, Centric


class ObjectMap:
    def __init__(self, locations) -> None:
        map = {}
        self.objects = []
        for obj in locations:
            self.objects.append(obj['name'])
            map[obj['name']] = [obj['location'][0], -obj['location'][2]]
        self.objects.append('camera')
        map['camera'] = [0.0, 0.0]
        
        self.yaw_history = [(0, 1)]
        self.map_history = [map]
        self.n_steps = 1
    
    
    def get_location(self, object, time_step=-1):
        return self.map_history[time_step][object]
    
    
    def get_distance(self, obj1, obj2, time_step=-1):
        loc1 = self.get_location(obj1, time_step=time_step)
        loc2 = self.get_location(obj2, time_step=time_step)
        
        distance = math.dist(loc1, loc2)
        rounded_distance = round(distance, 1)
        
        return rounded_distance
    
    
    def get_direction(self, target_obj, view_obj, time_step=-1):
        loc1 = self.get_location(target_obj, time_step=time_step)
        loc2 = self.get_location(view_obj, time_step=time_step)
        relative_x = loc1[0] - loc2[0]
        relative_y = loc1[1] - loc2[1]
        angle_radians = math.atan2(relative_y, relative_x)
        angle_degrees = math.degrees(angle_radians)
        adjusted_angle = 90 - angle_degrees
        
        if view_obj == 'camera':
            yaw = self.yaw_history[time_step]
            if yaw == (0, 1):
                view_yaw_deg = 0
            elif yaw == (1, 0):
                view_yaw_deg = 90
            elif yaw == (0, -1):
                view_yaw_deg = 180
            else:
                view_yaw_deg = 270
                
            adjusted_angle = (adjusted_angle - view_yaw_deg + 360) % 360
            index = int((adjusted_angle + 22.5) / 45) % 8
            return EGO_DIRECTIONS[index]
            
        else:
            adjusted_angle = (adjusted_angle + 360) % 360

            index = int((adjusted_angle + 22.5) / 45) % 8
            return ALLO_DIRECTIONS[index]
    
    
        
    def apply_actions(self, actions):
        if not actions:
            return 
        
        for action in actions:
            object = action.object
            location = self.get_location(object)
            direction = action.direction
            map = self.map_history[-1].copy()
            yaw = self.yaw_history[-1]
            
            if action.centric == Centric.EGO:
                dir = yaw
            else:
                dir = (0, 1)

            if direction == Direction.FORWARD:
                dx, dy = (dir[0], dir[1])
            if direction == Direction.BACKWARD:
                dx, dy = (-dir[0], -dir[1])
            elif direction == Direction.LEFT:
                dx, dy = (-dir[1], dir[0])
            else:
                dx, dy = (dir[1], -dir[0])
                
            if action.move_type == Movement.ROTATION:
                yaw = (dx, dy)
            else:
                location = [location[0] + dx * action.distance,
                            location[1] + dy * action.distance]
                map[object] = location
                
            self.yaw_history.append(yaw)
            self.map_history.append(map)
            self.n_steps += 1
            
        return 