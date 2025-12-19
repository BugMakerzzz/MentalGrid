#!/usr/bin/env bash

for type in distance_evaluation distance_location distance_tracking direction_evaluation direction_location direction_tracking
do 
    for centric in egocentric allocentric
    do 
        for n in 1 3 5
        do 
            python -m generate_data.generate_question --centric $centric --type $type --obj_steps $n --camera_steps $n
        done 
    done 
done 