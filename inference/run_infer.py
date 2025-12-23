#!/usr/bin/env python3
import os
import argparse
from .model import Model
from eval.load_data import load_dataset, save_json_data, load_remain_data
from eval.parse_result import extract_answer
from tqdm import tqdm
import random 
import yaml
from box import Box
from .tool_infer import ToolPipeline

random.seed(17)


def main(args):
    dataset = load_dataset(args=args, 
                        method=args.method,
                        split='val')
    
    config_path = f'configs/{args.method}/{args.model}.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = Box(yaml.load(f, Loader=yaml.FullLoader))
    
    output_dir = f'results/{args.dataset}/val/c{args.camera_steps}_o{args.obj_steps}_mr{args.move_ratio}_er{args.ego_ratio}/{args.type}_{args.centric}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 
                                f'{args.model}_{args.method}.json')
    
    if args.method in ['cot', 'map_cot']:
        model = Model(config=config.vlm)  
    else:
        pipeline = ToolPipeline(config=config)  
    
    # results, start_idx = load_last_index(output_path)
    results, dataset = load_remain_data(output_path, dataset, args.reload)
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Making inference"):
        if args.method in ['cot', 'map_cot']:
            response = model.generate(message=item['input'])
            history = None
        else:
            response, history = pipeline.generate(example=item)
        pred = extract_answer(response_text=response)
        answer = item['answer']
        cor_flag = pred == answer
        msg = {
            'id': item['id'],
            'image_path': item['image_path'],
            'prompt': item['sys_prompt'],
            'question': item['query'],
            'response': response,
            'pred': pred,
            'answer': answer,
            'cor_flag': cor_flag,
            'history': history
        }
        results.append(msg)
    # Save the updated dataset
        save_json_data(output_path, results)

    print(f"\nTesting complete. Updated dataset saved to {output_path}")
    # Load the dataset
   

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=17)
parser.add_argument('--method', type=str, default='cot')
parser.add_argument('--n_samples', type=int, default=200)
parser.add_argument('--dataset', type=str, default='replica')
parser.add_argument('--type', type=str, default='direction_evaluation')
parser.add_argument('--centric', type=str, default='allocentric')
parser.add_argument("--camera_steps", type=int, default=0)
parser.add_argument("--obj_steps", type=int, default=0)
parser.add_argument('--model', type=str, default='Qwen2.5-VL-7B')
parser.add_argument('--reload', action='store_true')
parser.add_argument("--move_ratio", type=float, default=0.6)
parser.add_argument("--ego_ratio", type=float, default=0.6)
args = parser.parse_args()
if __name__ == "__main__":
    main(args)

