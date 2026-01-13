#!/bin/bash

MODEL_NAMES=("Qwen2.5-VL-3B")
TASK=("VSI-Bench")
PROMPT_TYPE="cot"
DEBUG_SIZE=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_names)
      IFS=' ' read -r -a MODEL_NAMES <<< "$2"
      shift 2
      ;;
    --prompt_type)
      PROMPT_TYPE="$2"
      shift 2
      ;;
    --task)
      IFS=' ' read -r -a TASK <<< "$2"
      shift 2
      ;;
    --debug_mode)   
      DEBUG_MODE="--debug_mode"
      shift
      ;;
    --debug_size)  
      DEBUG_SIZE="$2"
      shift 2
      ;;
    --reload)     
      RELOAD="--reload"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Map model name to model path
# MODEL_CONFIG_DICT["qwenvl_7b"]="/models/Qwen/Qwen2.5-VL-7B-Instruct"

for model_name in "${MODEL_NAMES[@]}"; do

  for task_name in "${TASK[@]}"; do
    echo "Running evaluation: model=$model_name, task=$task_name"

    python -m eval.evaluator \
      --eval_task "$task_name" \
      --log_dir results/"${task_name}"/"${model_name}"/"${PROMPT_TYPE}" \
      --model_config "$model_name" \
      --prompt_type "$PROMPT_TYPE" \
      --num_frames 32 \
      --max_pixels $((512*28*28)) \
      --min_pixels $((16*28*28)) \
      --debug_size "$DEBUG_SIZE" \
      $DEBUG_MODE \
      $RELOAD
  done
done
