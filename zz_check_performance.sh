export CUDA_VISIBLE_DEVICES="0,1,3,4,5"

nohup python main.py \
	--model hf-causal-experimental \
	--model_args pretrained=SpassMedAI/SM70-MEDQA_MAlpaca100K_FT,load_in_8bit=True,use_accelerate=True\
	--tasks jmle,medqa_usmle,pubmedqa \
	--num_fewshot 5 \
	--device cuda \
    --batch_size 2 \
    --write_out \
    --output_base_path zz_results_jmle_pubmedqa_usmle_SM70-MEDQA_MAlpaca100K_FT_5s\
	> zz_logs_jmle_pubmedqa_usmle_SM70-MEDQA_MAlpaca100K_FT_5s.log 2>&1 &