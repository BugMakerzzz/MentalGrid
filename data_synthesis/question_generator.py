import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

from utils.template import QueryType, Centric, QUESTION_TEMPLATES, Movement, EGO_DIRECTIONS, ALLO_DIRECTIONS, TaskType, ACTION_PREFIX
from space.map import ObjectMap
from space.action import Action, Movement

import re

WRONG_OPTION_CNT = 4
MIN_DISTANCE = 0.5
BUCKETS = [0, 5, 15, 30, 45, 60, 90, 120, 150, 180]

def plot_toy_map(object_coords, save_path):

    plt.figure(figsize=(12, 8)) 
    
    all_x = [coord[0] for coord in object_coords.values()]
    all_y = [coord[1] for coord in object_coords.values()]
    
    texts = []
    for name, (x, y) in object_coords.items():
        # 如果是 agent，用特殊颜色标注
        color = 'red' if name.lower() == 'agent' else 'tab:blue'
        plt.scatter(x, y, s=120 if name.lower() == 'agent' else 100, c=color, zorder=5) 
        texts.append(plt.text(x, y, name, fontsize=12, zorder=10, fontweight='bold'))
    
    # --- 新增：Agent 朝向箭头 ---
    if 'agent' in object_coords:
        ax_pos = object_coords['agent']
        plt.arrow(ax_pos[0], ax_pos[1], 0, 0.6, 
                  width=0.02, 
                  color='red', 
                  head_width=0.1,       
                  head_length=0.12,   
                  zorder=5, 
                  length_includes_head=True)

    # 自动避让
    adjust_text(texts, 
                expand_text=(1.5, 1.8), 
                expand_points=(1.5, 1.5),
                force_text=(0.5, 1.0),
                arrowprops=dict(arrowstyle='-', color='red', lw=1.5, alpha=0.6))

    # --- 修改：方位标签与字体大小 ---
    plt.xlabel("East (X) →", fontsize=16, labelpad=10)
    plt.ylabel("North (Y) →", fontsize=16, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # 强制比例 1:1，保证北向是垂直向上的
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=0.1)

    if all_x and all_y:
        plt.xlim(min(all_x) - 1, max(all_x) + 1)
        plt.ylim(min(all_y) - 1, max(all_y) + 1)

    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()



def get_rotation_description(current_angle, previous_angle):

    # 1. Calculate the shortest angular distance
    delta = (current_angle - previous_angle + 180.0) % 360.0 - 180.0

    # In your system (90-angle), positive delta is Clockwise
    direction = "Clockwise" if delta > 0 else "Counter-clockwise"
    abs_delta = abs(delta)
    
    # nearest bucket
    closest = min(BUCKETS, key=lambda b: abs(b - abs_delta))

    if closest == 0:
        return "No noticeable rotation (about 0°)"
    
    # optional: make 180° less awkward
    if closest == 180:
        return "Turn-around (about 180°)"

    return f"{direction} rotation by about {closest}°"
    

def generate_rotation_negatives(correct_list):

    def get_opposite_dir(action):
        if "Clockwise" in action:
            return action.replace("Clockwise", "Counter-clockwise")
        elif "Counter-clockwise" in action:
            return action.replace("Counter-clockwise", "Clockwise")
        return action

    def get_neighbor_bucket(action):
        
        match = re.search(r'\d+', action)
        if not match:
            return [action]
        
        current_deg = int(match.group())
        if current_deg not in BUCKETS:
            return [action]
        
        idx = BUCKETS.index(current_deg)

        neighbors = []
        if idx > 1: 
            if idx == 2:
                neighbors.append("No noticeable rotation (about 0°)")
            else:
                neighbors.append(re.sub(r'\d+', str(BUCKETS[idx-2]), action))
        
        if idx < len(BUCKETS) - 2:
            if idx == len(BUCKETS) - 3: 
                neighbors.append("Turn-around (about 180°)")
            else:
                neighbors.append(re.sub(r'\d+', str(BUCKETS[idx+2]), action))

        return neighbors

    # 提取原始动作
    act1, act2 = correct_list[0], correct_list[1]
    act1_negatives = []
    act2_negatives = []

    act1_negatives.append(get_opposite_dir(act1))
    act1_negatives += get_neighbor_bucket(act1)

    act2_negatives.append(get_opposite_dir(act2))
    act2_negatives += get_neighbor_bucket(act2)
    
    w_answers = [f"{a}, {b}" for a in set(act1_negatives) for b in set(act2_negatives)
                    if a != act1 or b != act2]
    
    return w_answers

class QuestionGenerator:
    def __init__(self, task, args):
        master_rng = np.random.default_rng(args.seed)
        self.task = task
        self.query_rng, self.action_rng, self.answer_rng = master_rng.spawn(3)
        self.map = None

    def load_scene(self, scene_info):
        object_info = scene_info["object"]
        self.map = ObjectMap(object_info)
        
    
    def update_map(self, actions):
        self.map.apply_actions(actions)
    
        
    def generate_query(self, query_type):
        centric = self.query_rng.choice(list(Centric))
        
        templates = QUESTION_TEMPLATES[query_type][self.task][centric]
        template = self.query_rng.choice(templates)
        
        query_objects = list(self.query_rng.choice(self.map.objects, size=2, replace=False))
        
        attempts = 0
        while (self.map.get_distance(query_objects[0], query_objects[1]) < MIN_DISTANCE):
            query_objects = list(self.query_rng.choice(self.map.objects, size=2, replace=False))
            attempts += 1
            if attempts > 100:
                return None
            
        if '{view_obj}' not in template:
            query_objects[1] = 'agent'
        if '{target_obj}' not in template:
            query_objects[0] = 'agent'
        
        context = {
            "type": query_type,
            "centric": centric,
            "template": template,
            "target_obj": query_objects[0],
            "view_obj": query_objects[1]
        }
        
        if query_type in [QueryType.OBJ_DIR, QueryType.DIR_OBJ]:
            if centric == Centric.EGO:
                if '{target_obj}' in template:
                    context['meta'] = 'obj-obj'
                else:
                    context['meta'] = 'obj-cam'
            else:
                if '{view_obj}' in template:
                    context['meta'] = 'obj-obj'
                else:
                    context['meta'] = 'obj-cam'
        elif query_type in [QueryType.DIS_OBJ]:
            if templates.index(template) < 5:
                context['meta'] = 'max'
            else:
                context['meta'] = 'min'   
        else:
            pass
        
        return context

    
    def generate_answer(self, query_context):
        query_type = query_context['type']
        centric = query_context['centric']
        target_obj = query_context['target_obj']
        view_obj = query_context['view_obj']
        other_objects = [obj for obj in self.map.objects if obj not in [target_obj, view_obj]] 
        
        options = []
        masked_seq = None 
        
        if query_type == QueryType.OBJ_DIS:
            if self.task == TaskType.DPT:
                dists = [self.map.get_distance(obj1=view_obj, obj2=target_obj, time_step=i)
                    for i in range(self.map.n_steps)]
                
                golden_states = []
                for i in range(len(dists)-1):
                    diff = dists[i+1] - dists[i]
                    if diff > 0:
                        desc = f"+{diff:.1f}"
                    elif diff < 0:
                        desc = f"{diff:.1f}"
                    else:
                        desc = "0.0"
                    
                    golden_states.append(desc)

                
                length = len(golden_states) 
                pairs = [
                    (i, j) for i in range(length) for j in range(i + 1, length)
                    if abs(i - j) > 1 and (golden_states[i] != '0.0' or golden_states[j] != '0.0')
                ]
                
                if not pairs:
                    return None
                
                i, j = self.answer_rng.choice(pairs)
                
                masked = golden_states.copy()
                masked[i] = "[MASK]"
                masked[j] = "[MASK]"
                masked_seq = " -> ".join(masked)

                # options are ONLY the (mask_i, mask_j) fillings
                golden_answer = f"{golden_states[i]}, {golden_states[j]}"
            
            else:
                golden_answer = str(self.map.get_distance(obj1=target_obj, obj2=view_obj))
        
        elif query_type == QueryType.DIS_OBJ: 
            if self.task == TaskType.DPT:
                golden_states = []
                for i in range(self.map.n_steps):
                    dists = ({obj:self.map.get_distance(obj1=obj, obj2=target_obj, time_step=i) \
                        for obj in other_objects})
                    
                    if query_context['meta'] == 'max':
                        golden_answer = max(dists, key=dists.get)
                    else:
                        golden_answer = min(dists, key=dists.get)
                        
                    if not golden_states or golden_states[-1] != golden_answer:
                        golden_states.append(golden_answer)

                length = len(golden_states) 
                
                if length < 3:
                    return None
                
                pairs = [(i, j) for i in range(1, length) for j in range(i + 1, length)]
                i, j = self.answer_rng.choice(pairs)

                masked = golden_states.copy()
                masked[i] = "[MASK]"
                masked[j] = "[MASK]"
                masked_seq = " -> ".join(masked)
                
                golden_answer = f"{golden_states[i]}, {golden_states[j]}"
                
                w_answers = [
                    f"{a}, {b}" for a in other_objects for b in other_objects
                    if (a != golden_states[i]) or (b != golden_states[j])
                ]
                
                if len(w_answers) < WRONG_OPTION_CNT:
                    return None
                
                options = list(self.answer_rng.choice(w_answers, size=WRONG_OPTION_CNT, replace=False))
            else:
                dists = {obj:self.map.get_distance(obj1=obj, obj2=target_obj) for obj in other_objects}
                
                if query_context['meta'] == 'max':
                    golden_answer = max(dists, key=dists.get)
                else:
                    golden_answer = min(dists, key=dists.get)

                w_answers = [obj for obj in other_objects if obj != golden_answer]
                
                if len(w_answers) < WRONG_OPTION_CNT:
                    return None  
                
                options = list(self.answer_rng.choice(w_answers, size=WRONG_OPTION_CNT, replace=False))
        
        
        elif query_type == QueryType.OBJ_DIR:
            if self.task == TaskType.DPT:
                directions = [self.map.get_direction(target_obj=target_obj, view_obj=view_obj, 
                                    centric=centric, time_step=i, return_type='angle') for i in range(self.map.n_steps)]
                
                golden_states = []
                for i in range(len(directions)-1):
                    desc = get_rotation_description(directions[i+1], directions[i])
                    golden_states.append(desc)
                
                
                length = len(golden_states) 
                pairs = [
                    (i, j) for i in range(length) for j in range(i + 1, length)
                    if abs(i - j) > 1 and (golden_states[i] != 'No noticeable rotation' 
                                           or golden_states[j] != 'No noticeable rotation')
                ]
                
                if not pairs:
                    return None
                
                i, j = self.answer_rng.choice(pairs)

                masked = golden_states.copy()
                masked[i] = "[MASK]"
                masked[j] = "[MASK]"
                masked_seq = " -> ".join(masked)

                # options are ONLY the (mask_i, mask_j) fillings
                golden_answer = f"{golden_states[i]}, {golden_states[j]}"
                w_answers = generate_rotation_negatives(correct_list=[golden_states[i], golden_states[j]])
                
                if len(w_answers) < WRONG_OPTION_CNT:
                    return None 
                
                options = list(self.answer_rng.choice(w_answers, size=WRONG_OPTION_CNT, replace=False))
                
            else:
                golden_answer = self.map.get_direction(target_obj=target_obj, view_obj=view_obj, centric=centric)
                directions = EGO_DIRECTIONS if centric == Centric.EGO else ALLO_DIRECTIONS
                w_directions = [d for d in directions if d != golden_answer]
                options = list(self.answer_rng.choice(w_directions, size=WRONG_OPTION_CNT, replace=False))
              
        
        elif query_type == QueryType.DIR_OBJ:
            if self.task == TaskType.DPT:
                directions = EGO_DIRECTIONS if centric == Centric.EGO else ALLO_DIRECTIONS
            
                final_direction = ""
                obj_cnt = 0
                for direction in directions:
                    cnt = 0
                    last_objs = set()
                    for i in range(self.map.n_steps):   
                        obj_dir_map = {obj: self.map.get_direction(target_obj=obj, view_obj=target_obj, 
                                                            centric=centric, time_step=i)
                                    for obj in other_objects}
                        
                        cor_objs = set([obj for obj in obj_dir_map if obj_dir_map[obj] == direction])
                        if cor_objs - last_objs:
                            cnt += 1
                        last_objs = cor_objs
                    if cnt > obj_cnt:
                        obj_cnt = cnt
                        final_direction = direction
                
                if obj_cnt < 3:
                    return None 

                direction = final_direction
                query_context['direction'] = direction
                
                golden_states = []
                masked = []
                last_objs = set()
                for i in range(self.map.n_steps):   
                    obj_dir_map = {obj: self.map.get_direction(target_obj=obj, view_obj=target_obj, 
                                                            centric=centric, time_step=i)
                                    for obj in other_objects}
                        
                    cor_objs = set([obj for obj in obj_dir_map if obj_dir_map[obj] == direction])
                    if cor_objs - last_objs:
                        diff_objs = cor_objs - last_objs
                        golden_states.append(diff_objs)
                        masked.append(diff_objs.pop())
                    
                length = len(golden_states) 
                pairs = [(i, j) for i in range(1, length) for j in range(i+1, length)]
                i, j = self.answer_rng.choice(pairs)
                
                golden_answer = f"{masked[i]}, {masked[j]}"
                
                masked[i] = "[MASK]"
                masked[j] = "[MASK]"
                masked_seq = " -> ".join(masked)
                
                w_answers = [
                    f"{a}, {b}" for a in other_objects for b in other_objects
                    if (a not in golden_states[i]) or (b not in golden_states[j])
                ]
                
                if len(w_answers) < WRONG_OPTION_CNT:
                    return None
      
                options = list(self.answer_rng.choice(w_answers, size=WRONG_OPTION_CNT, replace=False))
                
            else:
                obj_dir_map = {obj: self.map.get_direction(target_obj=obj, view_obj=target_obj, 
                                    centric=centric) for obj in other_objects}
                directions = set(obj_dir_map.values())
                
                if len(directions) == 1:
                    return None
                
                direction = self.answer_rng.choice(list(directions))
                cor_objs = [obj for obj in obj_dir_map if obj_dir_map[obj] == direction]
                    
                query_context['direction'] = direction
                
                golden_answer = self.answer_rng.choice(cor_objs)
                
                w_answers = [obj for obj in other_objects if obj not in cor_objs]
              
                if len(w_answers) < WRONG_OPTION_CNT:
                    return None
                    
                options = list(self.answer_rng.choice(w_answers, size=WRONG_OPTION_CNT, replace=False))    
            
            
        else:
            raise(TypeError)
        
        if options:
            random_index = self.answer_rng.integers(low=0, high=len(options)+1)
            golden_option = chr(ord('A') + random_index)
            options.insert(random_index, golden_answer)
        else:
            golden_option = golden_answer    
            
        answer_context = {
            'golden_answer': golden_answer,
            'golden_option': golden_option,
            'options': options,
            "masked_seq": masked_seq
        }      
        
        return answer_context


    def generate_action(self, query_context, ego_ratio=0.5):
        query_objs = [query_context['target_obj'], query_context['view_obj']]
        
        if self.task == TaskType.DPT:
            min_step = 3
        else:
            min_step = 1
        max_step = 10
        
        if query_context['type'] in [QueryType.DIR_OBJ, QueryType.DIS_OBJ]:
            target_ratio = 0.7
        else:
            target_ratio = 1.0
            
        if query_context['centric'] == Centric.ALLO:
            obj_action_nums = self.action_rng.integers(min_step, max_step)
            agent_action_nums = 0
        else:
            pairs = [(i, j) for i in range(max_step + 1) for j in range(max_step + 1 - i) if i + j >= min_step]  
            agent_action_nums, obj_action_nums = self.action_rng.choice(pairs)
        
        obj_actions = []

        target_action_nums = math.ceil(obj_action_nums * target_ratio)
        target_objects = [o for o in query_objs if o != 'agent']
        
        for _ in range(obj_action_nums):
            action_centric = self.action_rng.choice(list(Centric), p=[ego_ratio, 1-ego_ratio])
            
            if target_objects and len(obj_actions) < target_action_nums:
                obj = self.action_rng.choice(target_objects) 
            else:
                obj = self.action_rng.choice(self.map.objects)
                
            obj_actions.append(Action(object=obj, move_type=Movement.TRANSLATION, rng=self.action_rng, centric=action_centric))

        agent_actions = []
        if agent_action_nums == 1:
            move = Movement.TRANSLATION
            agent_actions.append(Action(object='agent', move_type=move, rng=self.action_rng))
        
        else:
            for _ in range(agent_action_nums):
                move = self.action_rng.choice(list(Movement))
                agent_actions.append(Action(object='agent', move_type=move, rng=self.action_rng))
        
        actions = obj_actions + agent_actions
        self.action_rng.shuffle(actions)

        step_texts = [action.to_str() for action in actions]
        if len(step_texts) == 1:
            action_text = step_texts[0]
        else:
            parts = []
            for idx, txt in enumerate(step_texts):
                parts.append(f"At step {idx+1}, {txt}")
            action_text = ' '.join(parts) + ' '
        
        action_text = self.action_rng.choice(ACTION_PREFIX) + action_text
        
        return action_text, actions
            
 
    def render_question(self, query_context, answer_context, action_text):
        if 'direction' in query_context:
            direction = query_context['direction']
        else:
            direction = None
        
        context = {
            'target_obj':query_context['target_obj'], 
            'view_obj':query_context['view_obj'],
            'direction':direction
        }

        if answer_context['options']:
            option_text = ""
            for i, option in enumerate(answer_context['options']):
                option_text += f"({chr(ord('A') + i)}) {option} "
                
            option_text = option_text.strip()
            context['options'] = option_text
            
        context['masked_seq'] = answer_context['masked_seq']
        context['action_text'] = action_text
        
        query_template = query_context['template']
        query = query_template.format(**context)

        
        return query
    
    
    