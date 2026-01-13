from utils.load_data import *
from utils.metrics import *
from utils.model import Model
from utils.prompt import POST_PROMPT_TEMPLATES, DEFAULT_SYSTEM_PROMPT, TOY_MAP_SYSTEM_PROMPT, HINT_COT_SYSTEM_PROMPT, MAP_COT_SYSTEM_PROMPT
from box import Box
from loguru import logger as eval_logger
import time
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import pandas as pd
import os
import yaml
import re
import json

def evaluate_mentalgrid(json_file, video_dir, model_name, output_dir, log_file, num_frames=32, 
                      max_pixels=512*28*28, min_pixels=16*28*28, debug=False, debug_size=12, 
                      params_dict=None, prompt_type="cot", reload=False):
    logger = setup_logger(log_file, params_dict)
    start_time_process = time.time()

    df = load_json_data(json_file)
    if debug:
        df = df[:debug_size]
        logger.info(f"Debug mode enabled, randomly processing {debug_size} samples.")

    logger.info(f"Shard size: {len(df)}")

    config_name = model_name.split('_')[0]
    config_path = f'configs/eval/cot/{config_name}.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = Box(yaml.load(f, Loader=yaml.FullLoader))
    config.vlm.model_name = model_name
    
    model = Model(config=config.vlm)

    result_file = os.path.join(output_dir, f"results.json")
    results, df = load_remain_data(result_file, df, reload=reload)
    
    prompt_template = POST_PROMPT_TEMPLATES.get(prompt_type, POST_PROMPT_TEMPLATES["default"])
    if prompt_type == 'toy_map':
        system_prompt = TOY_MAP_SYSTEM_PROMPT
    elif prompt_type == 'hint_cot':
        system_prompt = HINT_COT_SYSTEM_PROMPT
    elif prompt_type == 'map_cot':
        system_prompt = MAP_COT_SYSTEM_PROMPT
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    for _, row in tqdm(enumerate(df), total=len(df)):

        content = []
        
        if prompt_type == 'toy_map':
            image_path = f"{row['image_path'].replace('images', 'toy_images')}.png"
            img = encode_path_to_base64(image_path)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        
        elif prompt_type == 'map_cot':
            pass
        
        else:
            image_dir = row['image_path']
            if not os.path.exists(image_dir):
                print("Warning: image not found at: ", image_dir)
                continue
            
            files = [f for f in os.listdir(image_dir)]
            for file in natsorted(files):
                img = encode_path_to_base64(os.path.join(image_dir, file))
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        
        prompt_text = f"Question: {row['query']}\n" 
        
        if prompt_type in ['hint_cot', 'map_cot']:
            init_map = row['meta']['map_history'][0]
            filtered_dict = {
                k: [round(x, 1) for x in v] 
                for k, v in init_map.items() 
                if k.lower() in row['query'].lower()
            }

            map_str = json.dumps(filtered_dict)
            
            if prompt_type == 'hint_cot':
                prompt_text = f"Hint:{map_str}\n" + prompt_text
            else:
                prompt_text = f"Map:{map_str}\n" + prompt_text
        
        if not row['meta']['answer']['options']:
            prompt_text += "\n" + prompt_template["na_post_prompt"]  
        else:
            prompt_text += "\n" + prompt_template["mca_post_prompt"]
            
        content.append({"type": "text", "text": prompt_text})
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]  

        predicted_raw = model.generate(messages)

        result = {
            'id': row['id'],
            'scene': row['meta']['scene'],
            'question': row['query'],
            'question_type':  row['meta']['task'] + '-' + row['meta']['query_type'],
            'ground_truth': row['answer'],
            'predicted_raw': predicted_raw,
            'prompt': system_prompt + prompt_text
        }
        
        result = mentalgrid_process_results(result)
        results.append(result)
    # Save the updated dataset
        save_json_data(result_file, results)
    
    
    end_time_process = time.time()
    elapsed_time_process = end_time_process - start_time_process

    elapsed_time_process_formatted = format_time(elapsed_time_process)
    logger.info(f"Results saved to: {result_file}, time usage: {elapsed_time_process_formatted}")
    return result_file, elapsed_time_process

def get_float_list(s):
    return [x for x in re.findall(r'(-?\d+(?:\.\d+)?)', s)]

def mentalgrid_process_results(doc): 
    if 'object_to_distance' not in doc['question_type']:
        for key, value in METRICS_FOR_MCA.items():
            doc['predicted_answer'] = extract_answer_text(doc['predicted_raw'])
            doc['accuracy'] = eval(value)(fuzzy_matching(doc['predicted_answer']), doc['ground_truth'])
    else:
        for key, value in METRICS_FOR_NA.items():
            try:
                if ',' in doc['ground_truth']:
                    chunks = doc['predicted_raw'].split('<answer>')
                    if len(chunks) == 1:
                        doc['predicted_answer'] = 'None'
                        doc['accuracy'] = WORST_CASE_FOR_METRICS[key]
                    else: 
                        if len(chunks) == 2:
                            doc['predicted_answer'] = extract_answer_text(doc['predicted_raw'])
                        else:
                            doc['predicted_answer'] = chunks[1].split('</answer>')[0] + ',' + chunks[2].split('</answer>')[0]
                        
                        preds = get_float_list(doc['predicted_answer'])
                        ground_truth = get_float_list(doc['ground_truth'])
                        
                        if not preds:
                            doc['accuracy'] = WORST_CASE_FOR_METRICS[key]
                        
                        elif len(preds) == 1:
                            doc['accuracy'] = eval(value)(to_float(preds[0]), to_float(ground_truth[0])) / 2
                        
                        else:
                            acc_1 = eval(value)(to_float(preds[0]), to_float(ground_truth[0]))
                            acc_2 = eval(value)(to_float(preds[1]), to_float(ground_truth[1]))
                            doc['accuracy'] = (acc_1 + acc_2) / 2
                else:
                    doc['predicted_answer'] = extract_answer_text(doc['predicted_raw'])
                    doc['accuracy'] = eval(value)(to_float(fuzzy_matching_num(doc['predicted_answer'])), to_float(doc['ground_truth'])) # Use 'predicted_answer'
            except TypeError:
                doc['accuracy'] = WORST_CASE_FOR_METRICS[key]

    return doc 


def mentalgrid_eval(json_file_path):
    results = load_json_data(json_file_path)
    results_df = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results_df.groupby('question_type').groups.items():
        per_question_type = results_df.iloc[question_type_indexes]
        
        metric_value = per_question_type['accuracy'].mean()
        output[f"{question_type}_accuracy"] = metric_value
                
    for scene, scene_indexes in results_df.groupby('scene').groups.items():
        per_scene = results_df.iloc[scene_indexes]
        
        metric_value = per_scene['accuracy'].mean()
        output[f"{scene}_accuracy"] = metric_value
        
    categories = {
        'static_spatial_perception_accuracy': [
            'static_spatial_perception-object_to_distance_accuracy',
            'static_spatial_perception-distance_to_object_accuracy',
            'static_spatial_perception-object_to_direction_accuracy',
            'static_spatial_perception-direction_to_object_accuracy'
        ],
        'final_state_prediction_accuracy': [
            'final_state_prediction-object_to_distance_accuracy',
            'final_state_prediction-distance_to_object_accuracy',
            'final_state_prediction-object_to_direction_accuracy',
            'final_state_prediction-direction_to_object_accuracy'
        ],
        'dynamic_process_tracking_accuracy': [
            'dynamic_process_tracking-object_to_distance_accuracy',
            'dynamic_process_tracking-distance_to_object_accuracy',
            'dynamic_process_tracking-object_to_direction_accuracy',
            'dynamic_process_tracking-direction_to_object_accuracy'
        ]
    }

    for cat_name, sub_keys in categories.items():
        vals = [output.get(k) for k in sub_keys if k in output]
        if vals:
            output[cat_name] = sum(vals) / len(vals)
        else:
            output[cat_name] = 0.0
            
    output['overall_accuracy'] = results_df['accuracy'].mean()

    eval_logger.info(f"Evaluation results: {output}")
    
    return output

