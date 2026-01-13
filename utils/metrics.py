import re
from functools import partial
import os
import json
import pandas as pd
import numpy as np


MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_direction",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
    
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

QUESTION_TYPES_W_BBOX = [
    "object_size_estimation",
    "object_abs_distance",
    "object_rel_distance",
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_direction",
    "object_rel_distance",
    "obj_appearance_order",
]

QUESTION_TYPES_WO_BBOX = [
    "object_counting",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}


BOXED_PATTERN = r"\\boxed\{([A-H])\}"
ANSWER_PATTERN = r"<answer>\s*([A-H])\s*</answer>"  # answer pattern
SIMPLE_DOT_PATTERN = r"(?:^|[^A-Za-z])([A-H])\s*\."  # pattern with dots, no restriction on content after the dot
SIMPLE_PATTERN = r"(?:^|[^A-Za-z])([A-H])(?:$|[^A-Za-z])"  # pattern without dots

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    
    return abs(pred - target) / (abs(target) + 1e-9)

def mean_relative_accuracy(pred, target, start, end, interval):
    if pred is None or target is None:
        return 0.
    
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def fuzzy_matching_num(pred):

    pred = pred.strip().lower() 

    number_words = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
        'zero': '0', 'a': '1', 'an': '1'  
    }

    for word, digit in number_words.items():
        if re.search(r'\b' + word + r'\b', pred):  
            return digit  

    number_match = re.search(r'(-?\d+(?:\.\d+)?)', pred)
    if number_match:
        return number_match.group(1)  

    return "None"  


def fuzzy_matching(answer):
    answer = answer.strip()

    match = re.search(r'^[A-F]\.?$', answer.split(' ')[0].strip())
    if match:
        answer=match.group(0).rstrip('.').upper()
        answer=answer.strip()
        return answer
    
    # First try to find standard format answer ($\boxed{X}$)
    boxed_matches = list(re.finditer(BOXED_PATTERN, answer, re.IGNORECASE))
    if boxed_matches:
        # Use the last matched answer
        return boxed_matches[-1].group(1).upper()

    # Next look for answer pattern
    answer_matches = list(re.finditer(ANSWER_PATTERN, answer, re.IGNORECASE))
    if answer_matches:
        # Use the first matched answer
        return answer_matches[0].group(1).upper()

    # Finally look for single letters
    # First find those with dots
    dot_matches = list(re.finditer(SIMPLE_DOT_PATTERN, answer, re.IGNORECASE))
    if dot_matches:
        # Use the last match with dots
        return dot_matches[-1].group(1).upper()

    # Lastly find those without dots
    simple_matches = list(re.finditer(SIMPLE_PATTERN, answer, re.IGNORECASE))
    if simple_matches:
        # Use the last match without dots
        return simple_matches[-1].group(1).upper()

    # If nothing is found, return the original text (will result in 0 points in subsequent comparison)
    return answer.upper()


