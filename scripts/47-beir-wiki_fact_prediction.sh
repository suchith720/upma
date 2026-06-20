datasets="climate-fever dbpedia-entity fever nq"

for dset in $datasets
do
	echo $dset

	python scripts/45-beir_fact_prediction.py --datasets $dset --model_name nomic-ai/nomic-embed-text-v1 --model_type nomic --batch_size 4096 \
		--use_data_parallel --metric_dir /data/suchith/outputs/benchmarks/02-nomic_embed_text_v1/ --pred_suffix hotpotqa
done

