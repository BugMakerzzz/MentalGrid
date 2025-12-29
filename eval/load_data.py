import json 
import os
import base64
import random 
from utils.prompt import ZERO_SHOT_COT_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT, MAP_COT_SYSTEM_PROMPT, MAP_PREDICTION_SYSTEM_PROMPT
from tqdm import tqdm
from PIL import Image
random.seed(17)


def encode_image(image_path):
    """Encode a single image file"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def embed_openai_input(example, system_prompt):
    content = []
    if system_prompt in [ZERO_SHOT_COT_SYSTEM_PROMPT, MAP_PREDICTION_SYSTEM_PROMPT]:
        base64_image = encode_image(example['image_path'])
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })
        content.append({"type": "text", "text": example['query']})
    elif system_prompt in [MAP_COT_SYSTEM_PROMPT]:
        text = f"Map: {example['meta']['map']['map'][0]}\n"
        text += f"Question: {example['query']}"
        content.append({"type": "text", "text": text})
    else:
        pass 
    message=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]
    return message

def add_message(
    messages,
    role: str = "user",    # user, system
    text: str = None,
    image: Image.Image = None,
):
    '''
    Helper function to add a new message to the conversation
    '''
    if image is not None:
        new_message = {
            'role': role,
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': text}
            ]
        }
    else:
        new_message = {
            'role': role,
            'content': [
                {'type': 'text', 'text': text}
            ]
        }

    # Append the new message
    messages.append(new_message)

    return messages


def load_system_prompt(method):
    if method == 'cot':
        return ZERO_SHOT_COT_SYSTEM_PROMPT
    elif method == 'map_cot':
        return MAP_COT_SYSTEM_PROMPT
    elif method == 'map_prediction':
        return MAP_PREDICTION_SYSTEM_PROMPT
    else:
        return DEFAULT_SYSTEM_PROMPT
    

def load_dataset(args, method, split='val'):
    path = f'data/data/{args.dataset}/{split}/c{args.camera_steps}_o{args.obj_steps}_mr{args.move_ratio}_er{args.ego_ratio}/{args.type}.json'
    dataset = load_json_data(path)
    dataset = dataset[:args.n_samples]
    system_prompt = load_system_prompt(method)
    data = []
    for i, example in tqdm(enumerate(dataset), total=len(dataset), desc="Loading dataset"):
        message = embed_openai_input(example=example, system_prompt=system_prompt)
        example['input'] = message
        example['sys_prompt'] = system_prompt
        data.append(example)
    return data 

def load_json_data(path):
    if not os.path.exists(path):
        print(f'path {path} not exists')
        return None
    with open(path, 'r') as f:
        data = json.load(f)
        f.close()
    return data

def save_json_data(path, data, mode='w'):
    with open(path, mode) as f:
        json.dump(data, f, indent=4)
        f.close()
    return 

def load_last_index(filepath):
    if not os.path.exists(filepath):
        return [], 0 
    with open(filepath, "r") as f:
        data = json.load(f)
    return data, len(data) 

def load_remain_data(filepath, dataset, reload=False):
    if not os.path.exists(filepath) or reload:
        return [], dataset
    with open(filepath, "r") as f:
        result = json.load(f)
    ids = []
    new_result = []
    for item in result:
        # if item ['preds'] == []:
        if item ['response'] == [""]:
            continue
        ids.append(item['id'])
        new_result.append(item)
    remain_data = []
    for item in dataset:
        if item['id'] in ids:
            continue
        remain_data.append(item)
    return new_result, remain_data