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
               "sink", "tv_screen", "umbrella", "toothbrush", "plant"]
WRONG_OPTION_CNT = 4
MIN_DISTANCE = 0.25

def setup_independent_generators(main_seed, num_generators):
    master_rng = np.random.default_rng(main_seed)
    return master_rng.spawn(num_generators)


def load_dataset_scenes(args):
    info_dir = f'data/source/{args.dataset}/'
    image_dir = f'data/source/{args.dataset}/images/'
    if not info_dir:
        raise ValueError("No subdirectories found under the given path.")

    objects = []
    for file in os.listdir(info_dir):
        if file.endswith(".json"):
            file_path = os.path.join(info_dir, file)
            with open(file_path, 'r') as f:
                object = json.load(f)
                if object['split'] != args.split:
                    continue
                scene = ('_').join(file.split('_')[:-1])
                image_path = os.path.join(image_dir, f"{scene}.png")
                object['scene'] = scene
                object['image_path'] = image_path
                objects.append(object)
                f.close()
    return objects   


def generate_query(objects, q_type, centric, rng):
    template = rng.choice(QUESTION_TEMPLATES[q_type][centric])
    query_objects = list(rng.choice(objects, size=2, replace=False))
    while (math.dist(query_objects[0], query_objects[1]) < MIN_DISTANCE):
        query_objects = list(rng.choice(objects, size=2, replace=False))
    if '{view_obj}' not in template:
        query_objects[1] = 'camera'
    if '{target_obj}' not in template:
        query_objects[0] = 'camera'
    
    context = {
        "type": q_type,
        "centric": centric,
        "text": template,
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
    return context


def generate_actions(objects, query_context, n_steps, rng, target_ratio, ego_ratio):
    if n_steps == 0:
        return []
    
    query_objects = [query_context['target_obj'], query_context['view_obj']]
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
                action = Action(object='camera', move_type=Movement.ROTATION, rng=rng)
            else: 
                action = Action(object='camera', move_type=Movement.TRANSLATION, rng=rng)
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
            parts.append(f"At step {idx+1}, {txt}")
        action_text = ' '.join(parts) + ' '
        
    return action_text, actions


def generate_answer(query_context, obj_map, rng):
    q_type = query_context['type']
    centric = query_context['centric']
    target_obj = query_context['target_obj']
    view_obj = query_context['view_obj']
    objects = [obj for obj in obj_map.objects if obj not in ['camera', target_obj, view_obj]] 
    
    options = []
    masked_seq = ""
    if q_type == QueryType.DISTANCE_EVALUATION:
        golden_answer = obj_map.get_distance(obj1=target_obj, obj2=view_obj)
        numbers = [round(i * 0.1, 1) for i in range(1, 101)]
        wrong_distances = [num for num in numbers if abs(num - golden_answer) >= 1]
        options = []
        while len(options) < WRONG_OPTION_CNT:
            num = rng.choice(wrong_distances)
            if all(abs(num - n) >= 1 for n in options):
                options.append(num)
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
        
        L = len(changes) 
        pairs = [(i, j) for i in range(L) for j in range(i+1, L) if abs(i-j) > 1]
        pairs_no_ends = [(i, j) for (i, j) in pairs if not (i in (0, L-1) and j in (0, L-1))]
        i, j = (pairs_no_ends or pairs)[rng.integers(0, len(pairs_no_ends or pairs))]

        masked = changes.copy()
        masked[i] = "[MASK]"
        masked[j] = "[MASK]"
        masked_seq = " -> ".join(masked)

        # options are ONLY the (mask_i, mask_j) fillings
        golden_answer = f"{changes[i]}, {changes[j]}"
        w_answers = [(a, b) for a in DISTANCES for b in DISTANCES
                if (a != changes[i]) or (b != changes[j])]
        options = list(rng.choice(w_answers, size=WRONG_OPTION_CNT, replace=False))
        options = [(', ').join(option) for option in options]

    elif q_type == QueryType.DIRECTION_EVALUATION:
        golden_answer = obj_map.get_direction(target_obj=target_obj, view_obj=view_obj, centric=centric)
        directions = EGO_DIRECTIONS if centric == Centric.EGO else ALLO_DIRECTIONS
        w_directions = [d for d in directions if d != golden_answer]
        options = list(rng.choice(w_directions, size=WRONG_OPTION_CNT, replace=False))
    
    elif q_type == QueryType.DIRECTION_LOCATION: 
        direction = query_context['direction']
        dict = {obj: obj_map.get_direction(target_obj=obj, view_obj=target_obj, centric=centric)
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
        dir_history = [obj_map.get_direction(target_obj=target_obj, view_obj=view_obj, time_step=i, centric=centric)
                       for i in range(obj_map.n_steps)]
        directions = EGO_DIRECTIONS if centric == Centric.EGO else ALLO_DIRECTIONS
        
        L = len(dir_history) 
        pairs = [(i, j) for i in range(L) for j in range(i+1, L) if abs(i-j) > 1]
        pairs_no_ends = [(i, j) for (i, j) in pairs if not (i in (0, L-1) and j in (0, L-1))]
        i, j = (pairs_no_ends or pairs)[rng.integers(0, len(pairs_no_ends or pairs))]

        masked = dir_history.copy()
        masked[i] = "[MASK]"
        masked[j] = "[MASK]"
        masked_seq = " -> ".join(masked)

        # options are ONLY the (mask_i, mask_j) fillings
        golden_answer = f"{dir_history[i]}, {dir_history[j]}"
        w_answers = [(a, b) for a in directions for b in directions
                if (a != dir_history[i]) or (b != dir_history[j])]
        options = list(rng.choice(w_answers, size=WRONG_OPTION_CNT, replace=False))
        options = [(', ').join(option) for option in options]
        
    else:
        pass
    
    random_index = rng.integers(low=0, high=len(options)+1)
    golden_option = chr(ord('A') + random_index)
    options.insert(random_index, golden_answer)
    answer_msg = {
        'golden_answer': golden_answer,
        'golden_option': golden_option,
        'options': options,
        "masked_seq": masked_seq
    }      
    
    return answer_msg 


def generate_single_question(query_context, obj_map, action_text, rng):
    
    answer_msg = generate_answer(query_context=query_context,
                                obj_map=obj_map,
                                rng=rng) 
    
    option_text = ""
    for i, option in enumerate(answer_msg['options']):
        option_text += f"({chr(ord('A') + i)}) {option} "
    option_text = option_text.strip()
    query_context['options'] = option_text
    query_context['masked_seq'] = answer_msg['masked_seq']
        
    query = ""
    if action_text:
        query += "Suppose we perform the following movements in the image: "
        query += action_text
    query_text = query_context['text']
    query += query_text.format(**query_context)

    query_context['options'] = answer_msg['options']
    query_context['golden_answer'] = answer_msg['golden_answer']
    query_context['type'] = query_context['type'].value
    query_context['centric'] = query_context['centric'].value
    
    return query, answer_msg['golden_option'], query_context


def main(args):
    q_type = QueryType(args.type)
    questions = []
    scene_infos = load_dataset_scenes(args)
    data_rng, action_rng, question_rng = setup_independent_generators(main_seed=args.seed, num_generators=3)
    
    idx = 0
    for scene_info in tqdm(scene_infos):
        object_info = scene_info['objects']
        scene_name = scene_info['scene']
        image_path = scene_info['image_path']
        query_history = []
        for centric in list(Centric):
            for _ in range(args.n_samples):
                obj_map = ObjectMap(object_info)
                objects = [obj for obj in obj_map.objects if obj != 'camera']

                query_context = generate_query(objects=objects, 
                                                q_type=q_type, 
                                                centric=centric,
                                                rng=data_rng)
                
                if query_context in query_history:
                    continue
                else:
                    query_history.append(query_context)
                obj_actions = generate_actions(objects=objects, query_context=query_context, 
                                            n_steps=args.obj_steps, rng=action_rng, 
                                            target_ratio=args.move_ratio, ego_ratio=args.ego_ratio)
                camera_actions = generate_actions(objects=None,  query_context=query_context, 
                                                n_steps=args.camera_steps, rng=action_rng, 
                                                target_ratio=args.move_ratio, ego_ratio=args.ego_ratio)
                action_text, actions = merge_action_texts(obj_actions=obj_actions, 
                                                        camera_actions=camera_actions, rng=action_rng)
                obj_map.apply_actions(actions)
            
                query, answer, query_context = generate_single_question(query_context=query_context,
                                                                        obj_map=obj_map,
                                                                        action_text=action_text,
                                                                        rng=question_rng)
                
                
                msg = {
                    'id': f"{args.type.upper()}_{centric.upper()}_Q{idx+1}",
                    'scene': scene_name,
                    'centric': centric,
                    'obj_cnt': scene_info['obj_cnt'],
                    'image_path': image_path,
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
                idx += 1

    question_dir = f'data/data/{args.dataset}/{args.split}/c{args.camera_steps}_o{args.obj_steps}_mr{args.move_ratio}_er{args.ego_ratio}/'
    if not os.path.exists(question_dir):
        os.makedirs(question_dir)
    question_path = os.path.join(question_dir, f'{args.type}.json')
    
    data_rng.shuffle(questions)
    with open(question_path, 'w') as f:
        json.dump(questions, f, indent=4)
        f.close()
    return 


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='replica')
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--split", type=str, default='val')
parser.add_argument("--n_samples", type=int, default=2)
parser.add_argument("--type", type=str, default='direction_evaluation')
parser.add_argument("--camera_steps", type=int, default=0)
parser.add_argument("--obj_steps", type=int, default=0)
parser.add_argument("--move_ratio", type=float, default=0.5)
parser.add_argument("--ego_ratio", type=float, default=0.5)
args = parser.parse_args()

    
if __name__ == '__main__':
    
    main(args)