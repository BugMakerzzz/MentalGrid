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
    
    
    def get_direction_index(self, angle):

        angle = angle % 360
        
        # 0: 345-15, 1: 15-75, 2: 75-105, 3: 105-165, 4: 165-195, 5: 195-255, 6: 255-285, 7: 285-345
        if angle < 15 or angle >= 345: return 0  # North / Forward
        if angle < 75:  return 1                 # NE / Right-Forward
        if angle < 105: return 2                 # East / Right
        if angle < 165: return 3                 # SE / Right-Backward
        if angle < 195: return 4                 # South / Backward
        if angle < 255: return 5                 # SW / Left-Backward
        if angle < 285: return 6                 # West / Left
        return 7                                 # NW / Left-Forward
    
    
    def get_location(self, object, time_step=-1):
        return self.map_history[time_step][object]
    
    
    def get_distance(self, obj1, obj2, time_step=-1):
        loc1 = self.get_location(obj1, time_step=time_step)
        loc2 = self.get_location(obj2, time_step=time_step)
        
        distance = math.dist(loc1, loc2)
        rounded_distance = round(distance, 1)
        
        return rounded_distance
    
    
    def get_direction(self, target_obj, view_obj, time_step=-1, centric=Centric.ALLO):
        loc1 = self.get_location(target_obj, time_step=time_step)
        loc2 = self.get_location(view_obj, time_step=time_step)
        relative_x = loc1[0] - loc2[0]
        relative_y = loc1[1] - loc2[1]
        angle_radians = math.atan2(relative_y, relative_x)
        angle_degrees = math.degrees(angle_radians)
        adjusted_angle = 90 - angle_degrees
        
        if centric == Centric.EGO:
            yaw = self.yaw_history[time_step]
            yaw_map = {(0, 1): 0, (1, 0): 90, (0, -1): 180}
            view_yaw_deg = yaw_map.get(yaw, 270)
                
            adjusted_angle = (adjusted_angle - view_yaw_deg + 360) % 360
            index = self.get_direction_index(adjusted_angle)
            return EGO_DIRECTIONS[index]
            
        else:
            adjusted_angle = (adjusted_angle + 360) % 360

            index = self.get_direction_index(adjusted_angle)
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