#!/usr/bin/env bash
for type in distance_evaluation distance_location direction_evaluation direction_location
do 
    for centric in egocentric allocentric
    do 
        python -m generate_data.generate_question --centric $centric --type $type --obj_steps 0 --camera_steps 0
        python -m generate_data.generate_question --centric $centric --type $type --obj_steps 2 --camera_steps 1
        python -m generate_data.generate_question --centric $centric --type $type --obj_steps 3 --camera_steps 2
        python -m generate_data.generate_question --centric $centric --type $type --obj_steps 4 --camera_steps 3
    done 
done 
for type in distance_tracking direction_tracking
do 
    for centric in egocentric allocentric
    do 
        python -m generate_data.generate_question --centric $centric --type $type --obj_steps 2 --camera_steps 1
        python -m generate_data.generate_question --centric $centric --type $type --obj_steps 3 --camera_steps 2
        python -m generate_data.generate_question --centric $centric --type $type --obj_steps 4 --camera_steps 3
    done 
done 