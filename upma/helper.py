import os, torch, json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp, argparse

from xcai.misc import *
from xcai.basics import *

from tqdm.auto import tqdm
from xcai.sdata import SMainXCDataset, SXCDataset

from typing import Optional, List

def early_fusion_beir_inference(output_dir:str, input_args:argparse.ArgumentParser, mname:str, data_file_format:str, 
                                linker_name:str, datasets:Optional[List]=None, metric_dir_name:Optional[str]="metrics", 
                                pred_dir_name:Optional[str]=None, eval_batch_size:Optional[int]=1600, 
                                ignore_metadata:Optional[bool]=False):
    
    metric_dir = f"{output_dir}/{metric_dir_name}"
    os.makedirs(metric_dir, exist_ok=True)

    input_args.only_test = input_args.do_test_inference = input_args.save_test_prediction = True

    datasets = BEIR_DATASETS if datasets is None else datasets
    for dataset in tqdm(datasets): 
        print(dataset)

        config_file = f"/data/datasets/beir/{dataset}/XC/configs/data.json"
        train_dset, test_dset = load_early_fusion_block(dataset, config_file, input_args)

        dataset_tag = dataset.replace("/", "-")

        data_file = data_file_format.format(dataset=dataset)
        if os.path.exists(data_file) and (not ignore_metadata):
            data_info = load_info(f"{input_args.pickle_dir}/{linker_name}/{dataset_tag}.joblib",
                                  data_file, mname, sequence_length=512)
            test_dset = SXCDataset(SMainXCDataset(data_info=data_info, data_lbl=test_dset.data.data_lbl, lbl_info=test_dset.data.lbl_info, 
                                                  return_scores=True))

        elif dataset != "quora" and (not ignore_metadata):
            print(f"WARNING: Missing raw file at {data_file}. Dataset '{dataset}' will be skipped.")
            continue

        input_args.prediction_suffix = dataset_tag
        trn_repr, tst_repr, lbl_repr, trn_pred, tst_pred, trn_metric, tst_metric = early_fusion_run(output_dir, input_args, mname, 
                                                                                                    test_dset, train_dset, 
                                                                                                    save_dir_name=pred_dir_name, 
                                                                                                    eval_batch_size=eval_batch_size)
        with open(f"{metric_dir}/{dataset_tag}.json", "w") as file:
            json.dump({dataset: tst_metric}, file, indent=4)

    collate_beir_metrics(metric_dir)

