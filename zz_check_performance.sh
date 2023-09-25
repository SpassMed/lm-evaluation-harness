export CUDA_VISIBLE_DEVICES=0,1,3,4,5

python main.py \
	--model hf-causal \
	--model_args pretrained=SpassMedAI/SM70_MEDQA_FT\
	--tasks usmle_self_eval_step1,usmle_self_eval_step2,usmle_self_eval_step3 \
	--num_fewshot 5 \
    --batch_size 2 \
    --write_out \
    --output_base_path results_usmle_self_eval_llama_2_7b