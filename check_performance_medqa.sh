export CUDA_VISIBLE_DEVICES="0,1,2,3"


# "(pretrained=SpassMedAI/SM70_MEDQA_FT,load_in_8bit=False,use_accelerate=True), num_fewshot: 4, batch_size: 4"
# "|       Task       |Version| Metric |Value |   |Stderr|
# |------------------|------:|--------|-----:|---|-----:|
# |medqa_usmle_gbaker|      0|acc     |0.5648|±  |0.0139|
# |                  |       |acc_norm|0.5546|±  |0.0139|"

# hf-causal-experimental (pretrained=/root/anubhav/LLM/bhatti/source/llama_models_ft_vast/llama-2-PH70_MedAlpaca300_ft_wLora_vast,load_in_8bit=False,use_accelerate=True), limit: None, provide_description: False, num_fewshot: 5, batch_size: 4
# |   Task    |Version| Metric |Value |   |Stderr|
# |-----------|------:|--------|-----:|---|-----:|
# |medqa_usmle|      0|acc     |0.5954|±  |0.0138|
# |           |       |acc_norm|0.5899|±  |0.0138|

tasks=medqa_usmle_gbaker
# limit=100
nshots=5
nbatch=4
pmodel=/root/anubhav/LLM/bhatti/source/llama_models_ft_vast
model=llama-2-PH70_MedAlpaca300_ft_wLora_vast
pathmodel=$pmodel/$model
load_in_8bit=False
use_accelerate=True

nohup python main.py \
	--model hf-causal-experimental \
	--model_args pretrained=${pathmodel},load_in_8bit=${load_in_8bit},use_accelerate=${use_accelerate}\
	--tasks $tasks \
	--num_fewshot $nshots \
	--device cuda \
    --batch_size $nbatch \
    --write_out \
    --output_base_path /root/anubhav/LLM/lm_eval_vast/lm-evaluation-harness/results/${tasks}_${model}_FT_${nshots}s\
	> /root/anubhav/LLM/lm_eval_vast/lm-evaluation-harness/lm_eval/logs/${tasks}_${model}_FT_${nshots}s_03.log 2>&1 &