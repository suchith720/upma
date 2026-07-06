# Conflation code

python scripts/01_conflation.py --pred_file /data/datasets/msmarco/XC/entity-gpt_ngame_trn_X_Y.npz --lbl_file //data/datasets/msmarco/XC/raw_data/entity_gpt.raw.txt  --trn_file /data/datasets/msmarco/XC/entity_gpt_trn_X_Y.npz --tst_file /data/datasets/msmarco/XC/entity_gpt_tst_X_Y.npz --topk 3 --batch_size 1024 --min_thresh 2 --max_thresh 100 --freq_thresh 50 --score_thresh 25 --diff_thresh 0.1 --embed_file /data/outputs/mogicX/01-msmarco-gpt-entity-linker-001/predictions/label_repr_full.pth

python scripts/45-beir_fact_prediction.py --datasets hotpotqa --model_name nomic-ai/nomic-embed-text-v1 --model_type nomic --batch_size 512 --use_data_parallel --metric_dir /data/suchith/outputs/benchmarks/02-nomic_embed_text_v1/

python scripts/43-evaluate_beir.py --datasets all --model_name Qwen/Qwen3-Embedding-0.6B --model_type qwen --batch_size 512 --use_data_parallel --metric_dir /data/suchith/outputs/benchmarks/04-qwen_embedding_0.6B/

python scripts/48-evaluate_harnesslm_beir.py --datasets arguana --query_model_name /home/sasokan/suchith/pretrained_models/01-Qwen3-Embedding-0.6B_pruned_50 --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-4B --doc_model_type qwen --batch_size 256 --use_data_parallel --metric_dir /home/sasokan/suchith/pretrained_models/01-Qwen3-Embedding-0.6B_pruned_50

# Pruning

python scripts/48-evaluate_harnesslm_beir.py --datasets arguana --query_model_name /data/scratch/suchith/pruning/01-Qwen3-Embedding-0.6B_pruned_50/ --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-0.6B --doc_model_type qwen --batch_size 256 --use_data_parallel --metric_dir /data/scratch/suchith/pruning/01-Qwen3-Embedding-0.6B_pruned_50/ --query_prefix ""

python scripts/48-evaluate_harnesslm_beir.py --datasets arguana --query_model_name /data/scratch/suchith/pruning/02-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_0.6B_pruned_50/ --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-0.6B --doc_model_type qwen --batch_size 256 --use_data_parallel --metric_dir /data/scratch/suchith/pruning/02-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_0.6B_pruned_50/ --query_prefix ""

# Alignment

python scripts/48-evaluate_harnesslm_beir.py --datasets arguana --query_model_name /data/scratch/suchith/alignment/01-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_0.6B_HF/ --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-0.6B --doc_model_type qwen --batch_size 256 --use_data_parallel --metric_dir /data/scratch/suchith/alignment/01-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_0.6B_HF/ --query_prefix ""

python scripts/48-evaluate_harnesslm_beir.py --datasets arguana --query_model_name /home/sasokan/scratch/suchith/alignment/02-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_4B_HF/ --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-4B --doc_model_type qwen --batch_size 256 --use_data_parallel --metric_dir /home/sasokan/scratch/suchith/alignment/02-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_4B_HF/ --query_prefix ""

python scripts/48-evaluate_harnesslm_beir.py --datasets arguana --query_model_name /data/scratch/suchith/alignment/03-Alignment_Qwen3-Embedding-0.6B_pruned_50_pruned_50_no_prompt_to_0.6B_HF/ --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-0.6B --doc_model_type qwen --batch_size 256 --use_data_parallel --metric_dir /data/scratch/suchith/alignment/03-Alignment_Qwen3-Embedding-0.6B_pruned_50_pruned_50_no_prompt_to_0.6B_HF/ --query_prefix ""

python scripts/48-evaluate_harnesslm_beir.py --datasets arguana --query_model_name /home/sasokan/scratch/suchith/alignment/04-Alignment_Qwen3-Embedding-0.6B_pruned_50_pruned_50_pruned_50_no_prompt_to_0.6B_HF/ --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-0.6B --doc_model_type qwen --batch_size 256 --use_data_parallel --metric_dir /home/sasokan/scratch/suchith/alignment/04-Alignment_Qwen3-Embedding-0.6B_pruned_50_pruned_50_pruned_50_no_prompt_to_0.6B_HF/ --query_prefix ""

python scripts/49-verify_harnesslm_beir.py --query_file /home/sasokan/suchith/datasets/nomic/train.tsv --no_header --student_model_name /data/scratch/suchith/alignment/03-Alignment_Qwen3-Embedding-0.6B_pruned_50_pruned_50_no_prompt_to_0.6B_HF/ --student_model_type qwen --teacher_model_name Qwen/Qwen3-Embedding-0.6B --teacher_model_type qwen --student_query_prefix "" --use_data_parallel

python scripts/49-verify_harnesslm_beir.py --query_file /home/sasokan/suchith/datasets/nomic/train.tsv --no_header --student_model_name /data/scratch/suchith/alignment/01-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_0.6B_0-500_HF/ --student_model_type qwen --teacher_model_name Qwen/Qwen3-Embedding-0.6B --teacher_model_type qwen --student_query_prefix "" --use_data_parallel

python scripts/50-harnesslm_fact_prediction.py --datasets all --query_model_name /data/outputs/reform/alignment/03-Alignment_Qwen3-Embedding-0.6B_pruned_50_pruned_50_no_prompt_to_0.6B_HF/ --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-0.6B --doc_model_type qwen --batch_size 256 --use_data_parallel --metric_dir /data/outputs/reform/alignment/03-Alignment_Qwen3-Embedding-0.6B_pruned_50_pruned_50_no_prompt_to_0.6B_HF/ --query_prefix ""

python scripts/48-evaluate_harnesslm_beir.py --datasets arguana --query_model_name /data/scratch/suchith/alignment/02-Alignment_1024/01-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_0.6B_HF/ --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-0.6B --doc_model_type qwen --batch_size 256 --use_data_parallel --metric_dir /data/scratch/suchith/alignment/02-Alignment_1024/01-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_0.6B_HF/ --query_prefix ""

python scripts/48-evaluate_harnesslm_beir.py --datasets all --query_model_name /data/scratch/suchith/alignment/02-Alignment_1024/01-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_0.6B_HF/ --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-0.6B --doc_model_type qwen --batch_size 256 --use_data_parallel --metric_dir /data/scratch/suchith/alignment/02-Alignment_1024/01-Alignment_Qwen3-Embedding-0.6B_pruned_50_no_prompt_to_0.6B_HF/ --query_prefix "" --embeddings_dir /data/outputs//benchmarks/04-qwen_embedding_0.6B/corpus_embeddings/

