#!/usr/bin/env bash

for type in distance_evaluation distance_location distance_tracking direction_evaluation direction_location direction_tracking
# for type in distance_evaluation
do 
    for centric in egocentric allocentric
    do 
        for n in 0 1 3 5
        do 
            python -m inference.run_infer --centric $centric --type $type --obj_steps $n --camera_steps $n --model $1 --method $2
        done 
    done 
done 