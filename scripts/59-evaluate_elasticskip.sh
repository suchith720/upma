models="01_spec-pf-keep14 02_hlm-pf-keep14 03_elastic_execL14"

for model in $models
do
	data_dir=/home/sasokan/suchith/elasticskip/dsdistil-e23-promptconsistency/_models/0.6B/$model

	python scripts/58-evaluate_asymmetric_beir_mteb.py --datasets all --query_model_name $data_dir --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-0.6B --doc_model_type qwen --embeddings_dir /data/suchith/outputs/benchmarks/07-qwen_embedding_0.6B/corpus_embeddings/ --multi_process --dtype bf16 --batch_size 256 --query_prefix "" --metric_dir $data_dir 
done

python scripts/58-evaluate_asymmetric_beir_mteb.py --datasets all --query_model_name Qwen/Qwen3-Embedding-0.6B --query_model_type qwen --doc_model_name Qwen/Qwen3-Embedding-0.6B --doc_model_type qwen --embeddings_dir /data/suchith/outputs/benchmarks/07-qwen_embedding_0.6B/corpus_embeddings/ --multi_process --dtype bf16 --batch_size 256 --query_prefix "" --metric_dir /data/suchith/outputs/benchmarks/08-qwen_embedding_0.6B_bare/


