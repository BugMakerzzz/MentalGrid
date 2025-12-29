import argparse
import os
from eval.load_data import load_json_data, save_json_data
from tqdm import tqdm 
from utils.prompt import DEFAULT_SYSTEM_PROMPT, ZERO_SHOT_COT_SYSTEM_PROMPT

def render_lf_format(instruction, response, system_prompt, image_path):
    return {
        "conversations": [
            {"from": "human", "value": f"<image>{instruction}"},
            {"from": "gpt", "value": response}
        ],
        "system": system_prompt,
        "images": [image_path]
    }


def load_raw_data(args):
    if args.type == 'all':
        data = []
        if args.camera_steps == -1:
            root_dir = f'data/data/{args.dataset}/train/'
        else:
            root_dir = f'data/data/{args.dataset}/train/c{args.camera_steps}_o{args.obj_steps}_mr{args.move_ratio}_er{args.ego_ratio}/'
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(".json"):
                    file_path = os.path.join(root, file)
                    json_data = load_json_data(file_path)
                    data += json_data
        return data
    else:
        path =f'data/data/{args.dataset}/train/c{args.camera_steps}_o{args.obj_steps}_mr{args.move_ratio}_er{args.ego_ratio}/{args.type}.json'
        return load_json_data(path)



def main(args):
    raw_data = load_raw_data(args=args)
    train_data = []
    
    if args.method == 'direct':
        system_prompt = DEFAULT_SYSTEM_PROMPT
    else:
        system_prompt = ZERO_SHOT_COT_SYSTEM_PROMPT
    
    for item in tqdm(raw_data):
        image_path = os.path.join(os.getcwd(), item['image_path'])
        instruction = item['query']
        if args.method == 'direct':
            response = f"# Answer: {item['answer']}"
        else:
            #TODO
            response = f"# Answer: {item['answer']}"
        msg = render_lf_format(instruction, response, system_prompt, image_path)
        train_data.append(msg)

    train_data_path = f'/mnt/userdata/ljc/code/LLaMA-Factory/data/mental_grid_l{args.camera_steps+args.obj_steps}_{args.method}.json'
    save_json_data(train_data_path, train_data)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='replica')
parser.add_argument("--type", type=str, default='all')
parser.add_argument("--camera_steps", type=int, default=0)
parser.add_argument("--obj_steps", type=int, default=0)
parser.add_argument("--move_ratio", type=float, default=0.5)
parser.add_argument("--ego_ratio", type=float, default=0.5)
parser.add_argument("--method", type=str, default='direct')
parser.add_argument("--append", action='store_true')
args = parser.parse_args()

    
if __name__ == '__main__':
    main(args)