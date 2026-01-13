import os
from datasets import Dataset
import argparse
from qwen_vl_utils import process_vision_info
from natsort import natsorted
from enum import Enum
import json
from utils.prompt import *
from utils.load_data import load_json_data, save_json_data
from utils.template import TaskType
from tqdm import tqdm

def rl_data_generator(file_path, data_ids):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if item['id'] not in data_ids:
                continue

            meta = item.pop('meta', None)
            item['task'] = meta['task']
            
            yield item


def generate_sft_data(args):
    path = f'data/MentalGrid/all/{args.task}/{args.split}.json'
    
    if args.split == 'train':
        data = load_json_data(path)
    else:
        if args.task == 'all':
            data = load_json_data(path)[:500]
        else:
            data = load_json_data(path)[:200]
    
    cot_path = f'results/{args.cot_type}/gpt-5.1/hint_cot/results.json'
    cot_data = load_json_data(cot_path)
    
    cot_dic = {}
    for item in cot_data:
        if item['accuracy'] > 0.7:
            cot = f"<think> {item['predicted_raw'].split('<answer>')[0].strip()} </think>"
            answer = f"<answer>{item['predicted_raw'].split('<answer>')[1]}"
            cot_dic[item['id']] = cot + '\n' + answer
    
    if args.split == 'train':
        if args.method == 'cot':
            rl_data_ids = [item['id'] for item in data if item['id'] in cot_dic]
        else:
            length = len(data) // 2
            rl_data_ids = [item['id'] for item in data[length:]]
            data = data[:length]
    else:
        rl_data_ids = [item['id'] for item in data]
        
    if args.method in ['direct', 'cot']:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    else:
        pass
    
    post_template = POST_PROMPT_TEMPLATES.get(args.method, 'default')
    
        
    train_data = []
    
    for item in tqdm(data):
        images = [f for f in os.listdir(item['image_path']) if f.endswith('.png')]
        root_dir = '/mnt/userdata/code/MentalGrid'
        image_paths = [os.path.join(root_dir, item['image_path'], f) for f in natsorted(images)]
        
        if item['answer'].isalpha():
            post_prompt = post_template['mca_post_prompt']
        else:
            post_prompt = post_template['na_post_prompt']
        
        if args.split == 'train':
            if args.method == 'cot':
                if item['id'] not in cot_dic:
                    continue
                response = cot_dic[item['id']]
            else:
                response = f"<answer>{item['answer']}</answer>."
        else:
            response = item['answer']
            
        img_str = '<image>' * len(image_paths)
        
        instruction = item['query'] + '\n' + post_prompt
        
        msg = {
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": f"{img_str}{instruction}", "role": "user"},
                {"content": response, "role": "assistant"}
            ],
            "system": system_prompt,
            "images": image_paths
        }
    
        train_data.append(msg)
    
    task_name = str(TaskType(args.task).name).lower() if args.task != 'all' else args.task
    
    if args.split == 'train':
        train_data_path = f'/mnt/userdata/ljc/code/LLaMA-Factory/data/mentalgrid_{task_name}_{args.method}_{args.split}.json'
    else:
        train_data_path = f'/mnt/userdata/ljc/code/LLaMA-Factory/data/mentalgrid_{task_name}_eval.json'
    
    save_json_data(train_data_path, train_data)
    
    return rl_data_ids


def generate_rl_data(args, rl_data_ids):
    data_path = f'data/MentalGrid/all/{args.task}/{args.split}.json'

    train_data = Dataset.from_generator(lambda: rl_data_generator(data_path, rl_data_ids))

    system_prompt = DEFAULT_SYSTEM_PROMPT
    
    instruction_following = (
        "Please think step by step to provide your detailed reasoning and then answer the question. "
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer MUST BE put in <answer> </answer> tags."
    )
    
    def make_map_fn(split):

        def process_fn(example, idx):
            question_id = example.get('id')
            question_raw = example.pop('query')
            # image = example.pop('image')[0]
            image_dir = example.pop('image_path')
            images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
            image_paths = [os.path.join(image_dir, f) for f in natsorted(images)]

            answer = example.pop('answer')

            assert all([os.path.exists(image_path) for image_path in image_paths]), f"Some images do not exist: {image_paths}"
            image_str = "<image>" * len(image_paths)
            
            mm_content = image_str + question_raw + instruction_following
            
            data = {
                "data_source": example.pop('task'),
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": mm_content,
                    }
                ],
                "images": [{"image": image_path} for image_path in image_paths],
                "ability": "visual_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'qid': question_id,
                    'images': [image_path for image_path in image_paths]
                }
            }
            return data

        return process_fn
    train_dataset = train_data.map(function=make_map_fn(args.split), with_indices=True, remove_columns=train_data.column_names, num_proc=32)
  
    print(train_dataset[0])
    print(len(train_dataset))
    local_dir = f'data/MentalGrid/'
    task_name = str(TaskType(args.task).name).lower() if args.task != 'all' else args.task
    
    train_dataset.to_parquet(os.path.join(local_dir, f'{args.split}_{task_name}.parquet'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--task", type=str, default='all')
    parser.add_argument("--cot_type", type=str, default='MentalGrid-SSP')
    parser.add_argument("--method", type=str, default='cot')
    parser.add_argument("--split", type=str, default='train')
    
    args = parser.parse_args()

    rl_data_ids = generate_sft_data(args)
    
    generate_rl_data(args, rl_data_ids)