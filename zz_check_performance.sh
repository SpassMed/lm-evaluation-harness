#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,3,4,5"

MODEL_NAME="meta-llama/Llama-2-70b-hf" #"SpassMedAI/BASE70_all-9_100K_FT"
PEFT_NAME=""

#begin postfix witha a hyphen '-'
POSTFIX="-jmle"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
TASKS="jmle"
N_FEW_SHOTS=3

# Convert slashes to hyphens for the MODEL_NAME and PEFT_NAME
SAFE_MODEL_NAME="${MODEL_NAME//\//-}${POSTFIX}"
SAFE_PEFT_NAME="${PEFT_NAME//\//-}"

# If PEFT_NAME is not empty, append it to SAFE_MODEL_NAME
if [[ ! -z "$PEFT_NAME" ]]; then
    SAFE_MODEL_NAME="${SAFE_MODEL_NAME}-peft-${SAFE_PEFT_NAME}${POSTFIX}"
fi

MODEL_ARGS="pretrained=${MODEL_NAME},load_in_8bit=True,use_accelerate=True"

# Append peft argument to MODEL_ARGS if PEFT_NAME is not empty
if [[ ! -z "$PEFT_NAME" ]]; then
    MODEL_ARGS="${MODEL_ARGS},peft=${PEFT_NAME}"
fi

nohup python main.py \
	--model hf-causal-experimental \
	--model_args ${MODEL_ARGS} \
	--tasks ${TASKS} \
	--num_fewshot ${N_FEW_SHOTS} \
	--device cuda \
    --batch_size 2 \
	--no_cache \
    --write_out \
    --output_base_path results/${TIMESTAMP}-${SAFE_MODEL_NAME}-${N_FEW_SHOTS}_shots \
	> logs/${TIMESTAMP}-${SAFE_MODEL_NAME}-${N_FEW_SHOTS}_shots.log 2>&1 &
