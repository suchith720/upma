import os, torch, json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp, argparse

from xcai.misc import *
from xcai.basics import *

from tqdm.auto import tqdm
from typing import Optional, Union, Callable, List

from xcai.core import *
from xcai.data import *
from xcai.sdata import SXCDataset, SMainXCDataset

from xcai.basics import *
from xcai.models.PPP0XX import DBT023, DBTConfig


DATASETS = [
    # "scifact",
    # "scidocs",
    # "msmarco",
    # "climate-fever",
    # "dbpedia-entity",
    # "fever",
    # "fiqa",
    # "hotpotqa",
    # "nfcorpus",
    # "nq",
    # "quora",
    # "webis-touche2020",
    # "trec-covid",
    "cqadupstack/android",
    "cqadupstack/english",
    "cqadupstack/gaming",
    "cqadupstack/gis",
    "cqadupstack/mathematica",
    "cqadupstack/physics",
    "cqadupstack/programmers",
    "cqadupstack/stats",
    "cqadupstack/tex",
    "cqadupstack/unix",
    "cqadupstack/webmasters",
    "cqadupstack/wordpress"
]


def run(output_dir:str, input_args:argparse.ArgumentParser, mname:str, test_dset:Union[XCDataset, SXCDataset],
        train_dset:Optional[Union[XCDataset, SXCDataset]]=None, collator:Optional[Callable]=identity_collate_fn, 
        save_dir_name:Optional[str]=None):

    args = XCLearningArguments(
        output_dir=output_dir,
        logging_first_step=True,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=1600,
        representation_num_beams=200,
        representation_accumulation_steps=10,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        save_total_limit=5,
        num_train_epochs=50,
        predict_with_representation=True,
        representation_search_type='BRUTEFORCE',
        search_normalize=False,

        adam_epsilon=1e-6,
        warmup_steps=1000,
        weight_decay=0.01,
        learning_rate=6e-5,
        label_names=['plbl2data_idx', 'plbl2data_data2ptr'],

        group_by_cluster=True,
        num_clustering_warmup_epochs=10,
        num_cluster_update_epochs=5,
        num_cluster_size_update_epochs=25,
        clustering_type='EXPO',
        minimum_cluster_size=2,
        maximum_cluster_size=1600,

        metric_for_best_model='P@1',
        load_best_model_at_end=True,
        target_indices_key='plbl2data_idx',
        target_pointer_key='plbl2data_data2ptr',

        use_encoder_parallel=True,
        max_grad_norm=None,
        fp16=True,

        use_cpu_for_searching=True,
        use_cpu_for_clustering=True,
    )

    config = DBTConfig(
        normalize = False,
        use_layer_norm = True,
        use_encoder_parallel = True,
    )

    def model_fn(mname, config):
        return DBT023.from_pretrained(mname, config=config)

    do_inference = check_inference_mode(input_args)
    model = load_model(args.output_dir, model_fn, {"mname": mname, "config": config}, do_inference=do_inference,
                       use_pretrained=input_args.use_pretrained)

    metric = PrecReclMrr(test_dset.data.n_lbl, test_dset.data.data_lbl_filterer, prop=None if train_dset is None else train_dset.data.data_lbl,
                         pk=10, rk=200, rep_pk=[1, 3, 5, 10], rep_rk=[10, 100, 200], mk=[5, 10, 20])

    learn = XCLearner(
        model=model,
        args=args,
        train_dataset=train_dset,
        eval_dataset=test_dset,
        data_collator=collator,
        compute_metrics=metric,
    )

    return main(learn, input_args, n_lbl=test_dset.data.n_lbl, eval_k=10, train_k=10, save_dir_name=save_dir_name)


def beir_inference(output_dir:str, input_args:argparse.ArgumentParser, mname:str, datasets:Optional[List]=None, 
                   metric_dir_name:Optional[str]="metrics", pred_dir_name:Optional[str]=None):
    
    metric_dir = f"{output_dir}/{metric_dir_name}"
    os.makedirs(metric_dir, exist_ok=True)

    input_args.only_test = input_args.do_test_inference = input_args.save_test_prediction = True

    datasets = BEIR_DATASETS if datasets is None else datasets
    for dataset in tqdm(datasets):
        print(dataset)

        config_file = f"/data/datasets/beir/{dataset}/XC/configs/data.json"
        train_dset, test_dset = load_early_fusion_block(dataset, config_file, input_args)

        dataset = dataset.replace("/", "-")

        input_args.prediction_suffix = dataset
        trn_repr, tst_repr, lbl_repr, trn_pred, tst_pred, trn_metric, tst_metric = run(output_dir, input_args, mname, test_dset, train_dset, 
                                                                                       save_dir_name=pred_dir_name)

        with open(f"{metric_dir}/{dataset}.json", "w") as file:
            json.dump({dataset: tst_metric}, file, indent=4)

    collate_beir_metrics(metric_dir)
        

if __name__ == '__main__':
    input_args = parse_args()
    extra_args = additional_args()

    output_dir = "/data/outputs/mogicX/37_training-msmarco-distilbert-from-scratch-008/"

    input_args.use_sxc_sampler = True
    input_args.pickle_dir = "/home/aiscuser/scratch1/datasets/processed/"
    mname = "distilbert-base-uncased"

    metric_dir_name, pred_dir_name = "metrics", "predictions"
    beir_inference(output_dir, input_args, mname, metric_dir_name=metric_dir_name, pred_dir_name=pred_dir_name, datasets=DATASETS)

