import numpy as np
import json 
from utils.template import QueryType, Centric, QUESTION_TEMPLATES, Movement, EGO_DIRECTIONS, ALLO_DIRECTIONS
import argparse
import os
from space.map import ObjectMap
from space.action import Action
from tqdm import tqdm
import math


DISTANCES = ["closer", "unchanged", "farther"]
ALL_OBJECTS = ["backpack", "candle", "coaster", "lamp", "picture", 
               "sink", "tv_screen", "umbrella", "toothbrush", "indoor_plant"]
WRONG_OPTION_CNT = 4

def setup_independent_generators(main_seed, num_generators):
    master_rng = np.random.default_rng(main_seed)
    return master_rng.spawn(num_generators)



def load_objects(root_path, obj_density, rng):
    dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

    if not dirs:
        raise ValueError("No subdirectories found under the given path.")

    scene_path = os.path.join(root_path, rng.choice(dirs))
    index = rng.integers(low=1, high=11)
    img_path = os.path.join(scene_path, f'{obj_density}_obj', 'images', f'scene{index}.png')
    obj_path = os.path.join(scene_path, f'{obj_density}_obj', f'scene{index}.json')
    
    while not os.path.exists(obj_path):
        scene_path = os.path.join(root_path, rng.choice(dirs))
        index = rng.integers(low=1, high=11)
        img_path = os.path.join(scene_path,  f'{obj_density}_obj', 'images', f'scene{index}.png')
        obj_path = os.path.join(scene_path,  f'{obj_density}_obj', f'scene{index}.json')

    with open(obj_path, 'r') as f:
        objects = json.load(f)['objects']
        f.close()
    
    return img_path, objects   

def sample_query_objects(objects, q_type, centric, rng):
    query_objects = list(rng.choice(objects, size=2, replace=False))
    if centric == Centric.EGO:
        query_objects[1] = 'camera'
        if q_type in [QueryType.DISTANCE_LOCATION, QueryType.DIRECTION_LOCATION]:
            query_objects[0] = 'camera'
    else:
        if q_type in [QueryType.DISTANCE_LOCATION, QueryType.DIRECTION_LOCATION]:
            query_objects[1] = 'camera'
    return query_objects


def generate_query(query_objects, q_type, centric, rng):
    template = rng.choice(QUESTION_TEMPLATES[q_type][centric])
    context = {
        "type": q_type,
        "centric": centric,
        "target_obj": query_objects[0],
        "view_obj": query_objects[1]
    }
    if q_type in [QueryType.DISTANCE_EVALUATION, 
                  QueryType.DISTANCE_TRACKING,
                  QueryType.DIRECTION_EVALUATION,
                  QueryType.DIRECTION_TRACKING]:
        pass 
    elif q_type == QueryType.DISTANCE_LOCATION:
        if 'far' in template:
            context['distance'] = 'far'
        else:
            context['distance'] = 'close'    
    elif q_type == QueryType.DIRECTION_LOCATION:
        if centric == Centric.EGO:
            direction = rng.choice(EGO_DIRECTIONS)
        else:
            direction = rng.choice(ALLO_DIRECTIONS) 
        context['direction'] = direction
    else:
        raise TypeError 
    return template, context


def generate_actions(objects, query_objects, n_steps, rng, target_ratio, ego_ratio):
    if n_steps == 0:
        return []
    
    actions = []
    target_steps = math.ceil(n_steps * target_ratio)
    if objects:
        target_objs = [obj for obj in query_objects if obj != 'camera']
        for _ in range(n_steps):
            action_centric = rng.choice(list(Centric), p=[ego_ratio, 1-ego_ratio])
            if len(actions) < target_steps and target_objs: 
                obj = rng.choice(target_objs)
            else:
                obj = rng.choice(objects)
            action = Action(object=obj, move_type=Movement.TRANSLATION, 
                            rng=rng, centric=action_centric)
            actions.append(action)   
    else:
        for _ in range(n_steps):
            if len(actions) < target_steps: 
                action = Action(object='camera', move_type=Movement.TRANSLATION, rng=rng)
            else: 
                action = Action(object='camera', move_type=Movement.ROTATION, rng=rng)
            actions.append(action)  
      
    return actions   
   
   
def merge_action_texts(obj_actions, camera_actions, rng):
    actions = obj_actions + camera_actions
    if not actions:
        return "", None
    rng.shuffle(actions)

    step_texts = [action.to_str() for action in actions]
    if len(step_texts) == 1:
        action_text = step_texts[0]
    else:
        parts = []
        for idx, txt in enumerate(step_texts):
            if idx == 0:
                parts.append(f"First, {txt}")
            elif idx == len(step_texts) - 1:
                parts.append(f"Finally, {txt} ")
            else:
                parts.append(f"Then, {txt}")
        action_text = ' '.join(parts)
        
    return action_text, actions


def generate_answer(query_context, obj_map, rng):
    q_type = query_context['type']
    centric = query_context['centric']
    target_obj = query_context['target_obj']
    view_obj = query_context['view_obj']
    objects = [obj for obj in obj_map.objects if obj not in ['camera', target_obj, view_obj]] 
    
    options = []
    if q_type == QueryType.DISTANCE_EVALUATION:
        golden_answer = obj_map.get_distance(obj1=target_obj, obj2=view_obj)
        numbers = [round(i * 0.1, 1) for i in range(1, 101)]
        wrong_distances = [num for num in numbers if abs(num - golden_answer) >= 1]
        options = []
        while len(options) < WRONG_OPTION_CNT:
            num = rng.choice(wrong_distances)
            if all(abs(num - n) >= 1 for n in options):  # 检查差值
                options.append(num)
        # while len(options) < WRONG_OPTION_CNT:
        #     if rng.random() < 0.5 and golden_answer - 1.0 > 0.1 or golden_answer + 1.0 > 10.0:
        #         val = rng.uniform(0.1, golden_answer - 1.0)
        #     else:
        #         val = rng.uniform(golden_answer + 1.0, 10.0)
        #     options.append(round(val, 1))
        golden_answer = f'{golden_answer} m'
        options = [f'{option} m' for option in options]
    
    elif q_type == QueryType.DISTANCE_LOCATION: 
        dists = {obj:obj_map.get_distance(obj1=obj, obj2=target_obj) for obj in objects}
        
        if query_context['distance'] == 'far':
            golden_answer = max(dists, key=dists.get)
        else:
            golden_answer = min(dists, key=dists.get)

        other_objs = [obj for obj in objects if obj != golden_answer]
        hallu_objs = [obj for obj in ALL_OBJECTS if obj not in obj_map.objects]
        if len(other_objs) < WRONG_OPTION_CNT:
            other_objs += rng.choice(hallu_objs, size=WRONG_OPTION_CNT-len(other_objs), replace=False).tolist()
            
        options = list(rng.choice(other_objs, size=WRONG_OPTION_CNT, replace=False))
    
    elif q_type == QueryType.DISTANCE_TRACKING:
        dists = [obj_map.get_distance(obj1=view_obj, obj2=target_obj, time_step=i) 
                 for i in range(obj_map.n_steps)]
        changes = []
        for i in range(1, len(dists)):
            if dists[i] == dists[i-1]:
                changes.append('unchanged')
            elif dists[i] > dists[i-1]:
                changes.append('farther')
            else:
                changes.append('closer')
                
        for i, cur in enumerate(changes):
            for alt in DISTANCES:
                if alt == cur:
                    continue
                cand = changes.copy()
                cand[i] = alt 
                options.append(cand)
        rng.shuffle(options)
        options = options[:min(len(options), WRONG_OPTION_CNT)]
        
        golden_answer = (' -> ').join(changes)
        options = [(' -> ').join(option) for option in options]

    elif q_type == QueryType.DIRECTION_EVALUATION:
        golden_answer = obj_map.get_direction(target_obj=target_obj, view_obj=view_obj)
        directions = EGO_DIRECTIONS if centric == Centric.EGO else ALLO_DIRECTIONS
        w_directions = [d for d in directions if d != golden_answer]
        options = list(rng.choice(w_directions, size=WRONG_OPTION_CNT, replace=False))
    
    elif q_type == QueryType.DIRECTION_LOCATION: 
        direction = query_context['direction']
        dict = {obj: obj_map.get_direction(target_obj=obj, view_obj=target_obj) 
                      for obj in objects}
        
        cor_objs = [obj for obj in dict if dict[obj] == direction]
        wro_objs = [obj for obj in objects if obj not in cor_objs]
        
        directions = EGO_DIRECTIONS if centric == Centric.EGO else ALLO_DIRECTIONS
        max_attempts = 100
        i = 0
        while not wro_objs or not cor_objs:
            direction = rng.choice(directions)
            cor_objs = [obj for obj in dict if dict[obj] == direction]
            wro_objs = [obj for obj in objects if obj not in cor_objs]
            i += 1
            if i > max_attempts:
                break

        query_context['direction'] = direction
        
        for i, cur in enumerate(cor_objs):
            for alt in wro_objs:
                cand = cor_objs.copy()
                cand[i] = alt  
                options.append(cand)
        rng.shuffle(options)
        options = options[:min(len(options), WRONG_OPTION_CNT)]
        
        golden_answer = (',').join(cor_objs)
        options = [(',').join(option) for option in options]
    

    elif q_type == QueryType.DIRECTION_TRACKING:
        dir_history = [obj_map.get_direction(target_obj=target_obj, view_obj=view_obj, time_step=i) 
                       for i in range(obj_map.n_steps)]
        directions = EGO_DIRECTIONS if centric == Centric.EGO else ALLO_DIRECTIONS
        
        for i, cur in enumerate(dir_history):
            for alt in directions:
                if alt == cur:
                    continue
                cand = dir_history.copy()
                cand[i] = alt 
                options.append(cand)
        rng.shuffle(options)
        options = options[:min(len(options), WRONG_OPTION_CNT)]
        
        golden_answer = (' -> ').join(directions)
        options = [(' -> ').join(option) for option in options]
        
    else:
        pass
    
    random_index = rng.integers(low=0, high=len(options)+1)
    golden_option = chr(ord('A') + random_index)
    options.insert(random_index, golden_answer)
    answer_msg = {
        'golden_answer': golden_answer,
        'golden_option': golden_option,
        'options': options
    }      
    
    return answer_msg 


def generate_single_question(q_type, centric, query_objects, obj_map, action_text, rng):
    query_text, query_context = generate_query(query_objects=query_objects, 
                                            q_type=q_type, 
                                            centric=centric,
                                            rng=rng)
    answer_msg = generate_answer(query_context=query_context,
                                obj_map=obj_map,
                                rng=rng) 
    
    option_text = ""
    for i, option in enumerate(answer_msg['options']):
        option_text += f"({chr(ord('A') + i)}) {option} "
    option_text = option_text.strip()
    query_context['options'] = option_text
        
    query = ""
    if action_text:
        query += "Suppose we perform the following movements in the image: "
        query += action_text
    query += query_text.format(**query_context)

    query_context['options'] = answer_msg['options']
    query_context['golden_answer'] = answer_msg['golden_answer']
    query_context['type'] = query_context['type'].value
    query_context['centric'] = query_context['centric'].value
    
    return query, answer_msg['golden_option'], query_context



def main(args):
    root_path = f'data/source/{args.dataset}/'
    q_type = QueryType(args.type)
    centric = Centric(args.centric)
    questions = []
    data_rng, action_rng, question_rng = setup_independent_generators(main_seed=args.seed, num_generators=3)
    for idx in tqdm(range(args.n_samples)):
        img_path, objects = load_objects(root_path, obj_density=args.obj_density, rng=data_rng)
        obj_map = ObjectMap(objects)
        objects = [obj for obj in obj_map.objects if obj != 'camera']
        
        query_objects = sample_query_objects(objects=objects, q_type=q_type, 
                                             centric=centric, rng=data_rng)
        
        obj_actions = generate_actions(objects=objects, query_objects=query_objects, 
                                       n_steps=args.obj_steps, rng=action_rng, 
                                       target_ratio=args.move_ratio, ego_ratio=args.ego_ratio)
        
        camera_actions = generate_actions(objects=None, query_objects=None, 
                                          n_steps=args.camera_steps, rng=action_rng, 
                                          target_ratio=args.move_ratio, ego_ratio=args.ego_ratio)
        
        action_text, actions = merge_action_texts(obj_actions=obj_actions, 
                                                  camera_actions=camera_actions, rng=action_rng)
        
        obj_map.apply_actions(actions)
        
        query, answer, query_context = generate_single_question(q_type=q_type,
                                                                centric=centric,
                                                                query_objects=query_objects, 
                                                                obj_map=obj_map,
                                                                action_text=action_text,
                                                                rng=question_rng)
        
        
        msg = {
            'id': f"{args.type.upper()}_{args.centric.upper()}_Q{idx+1}",
            'image_path': img_path,
            'query': query,
            'answer': answer,
            'context': {
                'query': query_context,
                'action': [action.to_dict() for action in actions] if actions else None,
                'map': {
                    'yam': obj_map.yaw_history,
                    'map': obj_map.map_history
                }
            }
        }
        
        questions.append(msg)

    question_dir = f'data/question/{args.dataset}/{args.obj_density}_obj/c{args.camera_steps}_o{args.obj_steps}/mr{args.move_ratio}_er{args.ego_ratio}/'
    if not os.path.exists(question_dir):
        os.makedirs(question_dir)
    
    question_path = os.path.join(question_dir, f'{args.type}_{args.centric}.json')
    with open(question_path, 'w') as f:
        json.dump(questions, f, indent=4)
        f.close()
    return 


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="replica")
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--n_samples", type=int, default=200)
parser.add_argument("--type", type=str, default='relative_direction')
parser.add_argument("--camera_steps", type=int, default=0)
parser.add_argument("--obj_steps", type=int, default=0)
parser.add_argument("--centric", type=str, default='allocentric')
parser.add_argument("--move_ratio", type=float, default=0.6)
parser.add_argument("--ego_ratio", type=float, default=0.6)
parser.add_argument("--obj_density", type=str, default='mid')
args = parser.parse_args()

    
if __name__ == '__main__':
    
    main(args)