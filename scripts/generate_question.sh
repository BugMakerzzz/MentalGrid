#!/usr/bin/env bash
for type in distance_evaluation distance_location direction_evaluation direction_location
do 
    python -m data_synthesis.generate_question --type $type --obj_steps 0 --camera_steps 0
    python -m data_synthesis.generate_question --type $type --obj_steps 2 --camera_steps 1
    python -m data_synthesis.generate_question --type $type --obj_steps 3 --camera_steps 2
    python -m data_synthesis.generate_question --type $type --obj_steps 4 --camera_steps 3
done 
for type in distance_tracking direction_tracking
do 
    python -m data_synthesis.generate_question --type $type --obj_steps 2 --camera_steps 1
    python -m data_synthesis.generate_question --type $type --obj_steps 3 --camera_steps 2
    python -m data_synthesis.generate_question --type $type --obj_steps 4 --camera_steps 3 
done 