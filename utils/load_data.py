import json 
import os
import base64
import random 
from PIL import Image
from io import BytesIO
import re
import numpy as np
from datetime import datetime, timedelta
import logging
from decord import VideoReader, cpu

random.seed(17)

def encode_path_to_base64(image_path):
    """Encode a single image file"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


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


def load_remain_data(filepath, data, reload=False):
    if not os.path.exists(filepath) or reload:
        return [], data
    
    with open(filepath, "r") as f:
        result = json.load(f)
    
    ids = []
    new_result = []
    for item in result:
        if item ['predicted_raw'] == "":
            continue
        ids.append(item['id'])
        new_result.append(item)
        
    remain_data = []
    for item in data:
        if item['id'] in ids:
            continue
        remain_data.append(item)
    return new_result, remain_data


def extract_answer_text(text_with_tags):
    match = re.search(r"<answer>(.*?)</answer>", text_with_tags, re.DOTALL)
    if match:
        return match.group(1).strip()  
    else:
        return "None"

def format_time(elapsed_seconds):
    time_delta = timedelta(seconds=int(elapsed_seconds))
    hours = time_delta.seconds // 3600
    minutes = (time_delta.seconds % 3600) // 60
    seconds = time_delta.seconds % 60
    return f"{hours:02}h{minutes:02}m{seconds:02}s"

def setup_logger(log_file, params_dict):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_with_timestamp = log_file.replace(".log", f"_{timestamp_str}.log")
    logging.basicConfig(
        filename=log_file_with_timestamp,
        level=logging.INFO,
        format=f'%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Running parameters:")
    for key, value in params_dict.items():
        logger.info(f"  {key}: {value}")
    return logger


def load_video_frames(video_path, num_frames=4, target_pixels=512*28*28):
    """Use decord to read video frames and return timestamps of those frames."""
    
    def calculate_target_size(original_width, original_height, target_pixels):
        """Calculate target size maintaining aspect ratio based on target pixels."""
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        
        target_height = int((target_pixels / aspect_ratio) ** 0.5)
        target_width = int((target_pixels * aspect_ratio) ** 0.5)
        
        return target_width, target_height
    
    def resize_image_to_target_pixels(image, target_pixels):
        """Resize image maintaining aspect ratio to approximate target pixels."""
        original_width, original_height = image.size
        target_width, target_height = calculate_target_size(original_width, original_height, target_pixels)
        
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    try:
        vr = VideoReader(video_path, ctx=cpu())
        total_frames = len(vr)
        video_duration = total_frames / vr.get_avg_fps() if vr.get_avg_fps() > 0 else total_frames / 30  # Estimate duration
        video_duration = int(video_duration)
        target_frames = num_frames
        
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frames_np = vr.get_batch(frame_indices).asnumpy()
        
        # Resize frames based on target_pixels
        frames_pil = [resize_image_to_target_pixels(Image.fromarray(f), target_pixels) for f in frames_np]
        
        timestamps = [int(idx / vr.get_avg_fps()) for idx in frame_indices] if vr.get_avg_fps() > 0 else [int(idx / 30) for idx in frame_indices]  # Get integer timestamps
        
        return frames_pil, timestamps, video_duration
    except Exception as e:
        return None, None, None
