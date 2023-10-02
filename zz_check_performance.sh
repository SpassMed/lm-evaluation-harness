export CUDA_VISIBLE_DEVICES="0,1,2,3"

nohup python main.py \
	--model hf-causal-experimental \
	--model_args pretrained=/root/anubhav/LLM/bhatti/source/llama_models_ft_vast/llama-2-70bMEDQA-pubmedqa_ft_wLora_vast,load_in_8bit=False,use_accelerate=True\
	--tasks pubmedqa_vast \
	--num_fewshot 5 \
	--device cuda \
    --batch_size 4 \
    --write_out \
    --output_base_path zz_results_jmle_pubmedqa_usmle_SM70-MEDQA_MAlpaca100K_FT_5s\
	> /root/anubhav/LLM/lm_eval_vast/lm-evaluation-harness/lm_eval/logs/zz_logs_pubmedqa_SM70-MEDQA_pubmedqa_FT_5s.log 2>&1 &