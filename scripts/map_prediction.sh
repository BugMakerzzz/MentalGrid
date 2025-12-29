#!/usr/bin/env bash
for type in distance_evaluation distance_location direction_evaluation direction_location
do 
    python -m inference.run_infer  --type $type --obj_steps 0 --camera_steps 0 --model $1 --method map_prediction --n_samples 100 --reload
done 