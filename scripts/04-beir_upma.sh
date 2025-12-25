#!/bin/bash

if [ $# -lt 1 ]
then
	echo "bash scripts/05-beir_linker.sh <expt_no>" 
	exit 1
fi

datasets="arguana msmarco climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scidocs scifact webis-touche2020 trec-covid \
	cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers \
	cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress"

result_file=results/03_upma-with-ngame-gpt-substring-linker-for-msmarco-$(printf "%03d" $1).txt
for dataset in $datasets
do
	echo $dataset
	suffix=$(echo $dataset | sed 's/\//-/g')
	
	echo $dataset >> $result_file
	CUDA_VISIBLE_DEVICES=0,1,2,3 python upma/03_upma-with-ngame-gpt-substring-linker-for-msmarco-beir-inference.py --dataset $dataset \
		--expt_no $1 --prediction_suffix $suffix >> $result_file
done

