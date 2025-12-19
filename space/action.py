from utils.template import Direction, Centric, Movement


class Action:
    def __init__(self, 
                 object: str, 
                 move_type: Movement, 
                 rng,
                 centric=Centric.EGO):
        self.object = object
        self.move_type = move_type
        self.centric = centric
        self.rng = rng
        self._generate_random_action()
        
    def _generate_random_action(self):
        if self.object == 'camera':
            if self.move_type == Movement.ROTATION:
                directions = [dir for dir in list(Direction) if dir != Direction.FORWARD]
                self.direction = self.rng.choice(directions)
                self.distance = None 
            # elif self.move_type == Movement.ALIGN:
            #     self.direction = self.rng.choice(directions)
            else:
                self.direction = Direction.FORWARD
                self.distance = self.rng.choice([0.5, 1.5, 2.5])
            
        else:
            directions = [dir for dir in list(Direction)]
            self.direction = self.rng.choice(directions)
            self.distance = self.rng.choice([0.5, 1.5, 2.5])
                

    def to_str(self):
        if self.move_type == Movement.ROTATION:
            map = {
                Direction.BACKWARD: "turn around",
                Direction.LEFT: "turn left",
                Direction.RIGHT: "turn right",
            }
        else:
            if self.centric == Centric.EGO: 
                if self.object == 'camera':
                    map = {
                        Direction.FORWARD: "move forward"
                    }
                else:
                    map = {
                        Direction.FORWARD: "moves forward",
                        Direction.BACKWARD: "moves backward",
                        Direction.LEFT: "moves to your left",
                        Direction.RIGHT: "moves to your right",
                    }
            else:
                map = {
                    Direction.FORWARD: "moves north",
                    Direction.BACKWARD: "moves south",
                    Direction.LEFT: "moves west",
                    Direction.RIGHT: "moves east",
                }

        if self.object == 'camera':
            if self.move_type == Movement.ROTATION:
                return f'you {map[self.direction]}.'
            else:
                return f'you {map[self.direction]} by {self.distance} meters.'
        else:
            return f"the {self.object} {map[self.direction]} by {self.distance} meters."


    def to_dict(self):
        
        return {
            'object': self.object,
            'type': self.move_type.value,
            'centric': self.centric.value,
            'direction': self.direction.value,
            'distance': self.distance           
        }
    
    