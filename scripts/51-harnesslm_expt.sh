# num="500 1000 1500 2000 2500"
num="2500"

for n in $num
do
	echo Experiment number: $n

	python scripts/48-evaluate_harnesslm_beir.py --datasets arguana --query_model_name /data/scratch/suchith/alignment/01-Alignment_128/01-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_0.6B_9-$n'_HF/' --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-0.6B --doc_model_type qwen --batch_size 256 --use_data_parallel --metric_dir /data/scratch/suchith/alignment/01-Alignment_128/01-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_0.6B_9-$n'_HF/' --query_prefix ""
	
	python scripts/49-verify_harnesslm_beir.py --query_file /home/sasokan/suchith/datasets/nomic/train.tsv --no_header --student_model_name /data/scratch/suchith/alignment/01-Alignment_128/01-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_0.6B_0-$n'_HF/' --student_model_type qwen --teacher_model_name Qwen/Qwen3-Embedding-0.6B --teacher_model_type qwen --student_query_prefix "" --use_data_parallel
done
