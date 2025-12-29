#!/usr/bin/env bash
for type in distance_evaluation distance_location direction_evaluation direction_location
do 

    python -m inference.run_infer --type $type --obj_steps 0 --camera_steps 0 --model $1 --method $2 --n_samples 100 --reload
done 

for type in distance_tracking direction_tracking distance_evaluation distance_location direction_evaluation direction_location
do 
    python -m inference.run_infer --type $type --obj_steps 2 --camera_steps 1 --model $1 --method $2 --n_samples 100
    python -m inference.run_infer --type $type --obj_steps 3 --camera_steps 2 --model $1 --method $2 --n_samples 100
done 