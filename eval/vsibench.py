from utils.metrics import *
from utils.load_data import *
from utils.prompt import POST_PROMPT_TEMPLATES
from utils.model import Model
from box import Box
from loguru import logger as eval_logger
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
import os
import yaml


def evaluate_vsibench(parquet_file, video_dir, model_name, output_dir, log_file, num_frames=32, 
                      max_pixels=512*28*28, min_pixels=16*28*28, debug=False, debug_size=12, 
                      params_dict=None, prompt_type="cot"):
    logger = setup_logger(log_file, params_dict)
    start_time_process = time.time()

    df = pd.read_parquet(parquet_file)
    if debug:
        df = df.sample(n=debug_size, random_state=17)
        logger.info(f"Debug mode enabled, randomly processing {debug_size} samples.")


    logger.info(f"Shard size: {len(df)}")

    config_name = model_name.split('_')[0]
    config_path = f'configs/eval/{prompt_type}/{config_name}.yml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = Box(yaml.load(f, Loader=yaml.FullLoader))
    config.vlm.model_name = model_name
    
    model = Model(config=config.vlm)

    results = []

    prompt_template = POST_PROMPT_TEMPLATES.get(prompt_type, POST_PROMPT_TEMPLATES["default"])
    
    for _, row in tqdm(df.iterrows(), total=len(df)):

        content = []
        
        video_path = os.path.join(video_dir, row['dataset'], f"{row['scene_name']}.mp4")
        if not os.path.exists(video_path):
            print("Warning: video not found at: ", video_path)
            continue
        
        frames, timestamps, duration = load_video_frames(video_path, num_frames, max_pixels)
        if frames is None:
            print("Warning: failed to extract frames for: ", video_path)
            continue
        
        for img in frames:
            b64 = encode_image_to_base64(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        

        question = row['question']
        options = row.get('options')
        if options is not None and len(options) > 0:
            options = options.tolist()                    
            question += "\nOptions:\n" + "\n".join(options)
            
        # Build prompt text

        prompt_text = prompt_template["pre_prompt"].format(question=question)
        
        if row['question_type'] in MCA_QUESTION_TYPES:
            prompt_text += "\n" + prompt_template["mca_post_prompt"]
        elif row['question_type'] in NA_QUESTION_TYPES:
            prompt_text += "\n" + prompt_template["na_post_prompt"]  
              
        content.append({"type": "text", "text": prompt_text})
        
        messages = [{"role": "user", "content": content}]  

        predicted_raw = model.generate(messages)

        result = {
            'id': row['id'],
            'dataset': row['dataset'],
            'scene_name': row['scene_name'],
            'question': row['question'],
            'ground_truth': row['ground_truth'],
            'predicted_answer': predicted_raw,
            'question_type': row['question_type'],
            'prompt': prompt_text,
            'frame_timestamps': timestamps, 
            'video_duration': duration
        }
        results.append(result)

    # Write results to file
    process_output_file = os.path.join(output_dir, f"VSI-Bench_results.json")
    save_json_data(process_output_file, results)

    end_time_process = time.time()
    elapsed_time_process = end_time_process - start_time_process

    elapsed_time_process_formatted = format_time(elapsed_time_process)
    logger.info(f"Results saved to: {process_output_file}, time usage: {elapsed_time_process_formatted}")
    return process_output_file, elapsed_time_process


def vsibench_aggregate_results(results):
    results_df = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results_df.groupby('question_type').groups.items():
        per_question_type = results_df.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                metric_value = per_question_type[metric].mean()
                output[f"{question_type}_{metric}"] = metric_value
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                metric_value = per_question_type[metric].mean()
                output[f"{question_type}_{metric}"] = metric_value
        else:
            raise ValueError(f"Unknown question type: {question_type}")
    
    # Calculate the combined metric for object_rel_direction
    if 'object_rel_direction_accuracy' not in output:
        try:
            output['object_rel_direction_accuracy'] = sum([
                output.pop('object_rel_direction_easy_accuracy'),
                output.pop('object_rel_direction_medium_accuracy'),
                output.pop('object_rel_direction_hard_accuracy'),
            ]) / 3.
        except:
            output['object_rel_direction_accuracy'] = 0
            
            
    # Store metrics for MCA and NA type questions separately
    mca_metrics = []
    na_metrics = []
    for key, metric_value in output.items():
        if '_accuracy' in key:
            question_type = key.replace('_accuracy', '')
        else:
            question_type = key.rsplit('_', 1)[0]
        
        if question_type in MCA_QUESTION_TYPES:
            mca_metrics.append(metric_value)
        elif question_type in NA_QUESTION_TYPES:
            na_metrics.append(metric_value)
    
    # Calculate overall accuracy as the average of all metrics
    output['overall_accuracy'] = sum(list(output.values())) / len(list(output.values()))
    eval_logger.info(f"Evaluation results: {output}")
    
    # Calculate mca accuracy and na accuracy
    output['mca_accuracy'] = sum(mca_metrics) / len(mca_metrics) if mca_metrics else 0
    output['na_accuracy'] = sum(na_metrics) / len(na_metrics) if na_metrics else 0
    
    return output


def vsi_process_results(doc): 
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc['predicted_answer']=doc['predicted_answer'].replace("Answer:","")
            doc[key] = eval(value)(fuzzy_matching(doc['predicted_answer']), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching_num(doc['predicted_answer'])), to_float(doc['ground_truth'])) # Use 'predicted_answer'
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return doc 


def vsibench_eval(json_file_path, mode="cot"):
    results = []
    data = load_json_data(json_file_path)
    for doc in data:
        if mode in ("cot") and "<answer>" in doc["predicted_answer"] :
            doc["predicted_answer"]=extract_answer_text(doc["predicted_answer"])
        processed_doc = vsi_process_results(doc)  # Process each doc to add metrics
        results.append(processed_doc)

    aggregated_results = vsibench_aggregate_results(results)  # Aggregate results after processing
    return aggregated_results