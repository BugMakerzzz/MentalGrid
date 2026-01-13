import random
import json 
import argparse
import os
import shutil
from tqdm import tqdm
from utils.load_data import save_json_data, load_json_data
from .question_generator import QuestionGenerator, plot_toy_map
from utils.template import TaskType, ALL_SCENES, QueryType
from space.map import ObjectMap
random.seed(17)


def load_scene_info(scene_ls, split):
    scene_infos = []
    for scene_name in scene_ls:
        info_dir = f'scenes/{scene_name}/{split}'
        old_image_dir = f'scenes/{scene_name}/{split}/images/'
        new_image_dir = f'data/MentalGrid/images/'
        toy_image_dir = f'data/MentalGrid/toy_images/'
        
        if not os.path.exists(info_dir):
            continue
        
        for file in tqdm(os.listdir(info_dir)):
            if file.endswith(".json"):
                file_path = os.path.join(info_dir, file)
                
                info = load_json_data(file_path)
                sub_scene = ('_').join(file.split('_')[:-1])
                
                old_image_path = os.path.join(old_image_dir, sub_scene)
                image_path = os.path.join(new_image_dir, sub_scene)
                toy_image_path = os.path.join(toy_image_dir, f'{sub_scene}.png')
                
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
            
                    for filename in os.listdir(old_image_path):
                        if filename.lower().endswith('.png'):
                            shutil.copy2(os.path.join(old_image_path, filename), image_path)
                
                if not os.path.exists(toy_image_path):
                    map = ObjectMap(info['object'])
                    plot_toy_map(object_coords=map.map_history[0], save_path=toy_image_path)
                
                info['scene'] = scene_name
                info['sub_scene'] = sub_scene
                info['image_path'] = image_path
                    
                scene_infos.append(info)
                
    return scene_infos
    
    
def main(args):

    scene_infos = load_scene_info(scene_ls=ALL_SCENES, split=args.split)
    
    all_questions = {scene: {'all':[]} for scene in ALL_SCENES}
    all_questions['all'] = {}
    all_questions['all']['all'] = []
    
    for task in TaskType:
        generator = QuestionGenerator(task=task, args=args)
        idx = 0 
        
        for k in all_questions:
            all_questions[k][task.value] = []
        
        for scene_info in tqdm(scene_infos):
            
            for query_type in QueryType:
                generator.load_scene(scene_info)
                
                q_ctx = generator.generate_query(query_type=query_type)
                
                if not q_ctx: 
                    continue
                
                if task == TaskType.SSP:
                    action_text = ""
                    actions = None

                else:
                    action_text, actions = generator.generate_action(query_context=q_ctx,
                                                                    ego_ratio=args.ego_ratio)
                    generator.update_map(actions)
                    
                ans_ctx = generator.generate_answer(q_ctx)
    
                if not ans_ctx: 
                    continue
                
                query_text = generator.render_question(q_ctx, ans_ctx, action_text)
                
                q_ctx['type'] = q_ctx['type'].value
                q_ctx['centric'] = q_ctx['centric'].value

                msg = {
                    'id': f"{task.value.upper()}_{query_type.value.upper()}_Q{idx+1}",
                    'image_path': scene_info['image_path'],
                    'query': query_text,
                    'answer': ans_ctx['golden_option'],
                    'meta': {
                        'query_type': query_type.value,
                        'task': task.value,
                        'scene': scene_info['scene'],
                        'sub_scene': scene_info['sub_scene'],
                        "image_cnt": scene_info['view_cnt'],
                        'query': q_ctx,
                        'action': [a.to_dict() for a in actions] if actions else None,
                        'answer': ans_ctx,
                        'yaw_history': generator.map.yaw_history, 
                        'map_history': generator.map.map_history
                    }
                }
                
                
                all_questions[scene_info['scene']][task.value].append(msg)
                all_questions[scene_info['scene']]['all'].append(msg)
                
                all_questions['all'][task.value].append(msg)
                all_questions['all']['all'].append(msg)
                
                idx += 1
                    

    for scene, v in all_questions.items():
        for task, data in v.items():
            random.shuffle(data)
                  
            question_dir = f'data/MentalGrid/{scene}/{task}/'
            if not os.path.exists(question_dir):
                os.makedirs(question_dir)
            
            path = os.path.join(question_dir, f'{args.split}.json')
            if not os.path.exists(path) or args.reload:
                save_json_data(path=path, data=data)
    


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--split", type=str, default='train')
parser.add_argument("--ego_ratio", type=float, default=0.5)
parser.add_argument("--reload", action='store_true')

args = parser.parse_args()


if __name__ == '__main__':
    main(args)
