import scipy.sparse as sp, numpy as np, os, json
from tqdm.auto import tqdm
from typing import Optional

import xclib.evaluation.xc_metrics as xm
from xcai.misc import *
from xcai.metrics import *


def beir_inference(batch_size:Optional[int]=1000):
    beir_metrics = dict()
    linker_dir = "/data/outputs/upma/00_msmarco-gpt-concept-substring-linker-with-ngame-loss-001/"

    for dataset in tqdm(BEIR_DATASETS):
        dataset_prefix = dataset.replace("/", "-")

        fname = f"{linker_dir}/cross_predictions/document-substring_sq-substring/test_predictions_{dataset_prefix}.npz"
        if not os.path.exists(fname): continue

        data_meta = sp.load_npz(fname)

        lbl_meta = sp.load_npz(f"/data/datasets/beir/{dataset}/XC/document_substring/lbl_sq-substring.npz")
        meta_lbl = lbl_meta.T

        pred_lbl = sp.vstack([data_meta[i:i+batch_size] @ meta_lbl for i in tqdm(range(0, data_meta.shape[0], batch_size))])
        data_lbl = sp.load_npz(f"/data/datasets/beir/{dataset}/XC/tst_X_Y.npz")

        metric = PrecReclMrr(data_lbl.shape[1], pk=10, rk=200, rep_pk=[1, 3, 5, 10], rep_rk=[10, 100, 200], mk=[5, 10, 20])
        beir_metrics[dataset_prefix] = {k:float(v) for k,v in metric.func(pred_lbl, data_lbl, **metric.kwargs).items()}

    metric_dir = "/data/outputs/upma/00_msmarco-gpt-concept-substring-linker-with-ngame-loss-001/cross_metrics/document-substring_sq-substring"
    os.makedirs(metric_dir, exist_ok=True)

    with open(f"{metric_dir}/beir.json", "w") as file:
        json.dump(beir_metrics, file, indent=4)


def multiply(data_meta, meta_lbl, batch_size:Optional[int]=1000):
    return sp.vstack([data_meta[i:i+batch_size] @ meta_lbl for i in tqdm(range(0, data_meta.shape[0], batch_size))]) 


def infer(data_meta_file:str, lbl_meta_file:str, data_lbl_file:str):
    data_meta = sp.load_npz(data_meta_file)
    meta_lbl = sp.load_npz(lbl_meta_file).T.tocsr()

    tst_pred_lbl = multiply(data_meta, meta_lbl) 

    data_lbl = sp.load_npz(data_lbl_file)
    metric = PrecReclMrr(data_lbl.shape[1], pk=10, rk=200, rep_pk=[1, 3, 5, 10], rep_rk=[10, 100, 200], mk=[5, 10, 20])

    return {k:float(v) for k,v in metric.func(tst_pred_lbl, data_lbl, **metric.kwargs).items()}


def gpt_inference(batch_size:Optional[int]=1000):
    metrics = {"msmarco": {}}

    for meta_type in ["substring", "narrow_substring", "intent_substring"]:
        m = metrics["msmarco"].setdefault(meta_type, {})

        # Test dataset
        m["test"] = infer(f"/data/datasets/beir/msmarco/XC/{meta_type}/all-substring_tst_X_Y.npz", 
                          f"/data/datasets/beir/msmarco/XC/{meta_type}/all-substring_lbl_X_Y.npz",
                          f"/data/datasets/beir/msmarco/XC/tst_X_Y.npz") 

        # Test dataset
        m["train"] = infer(f"/data/datasets/beir/msmarco/XC/{meta_type}/substring_trn_X_Y.npz", 
                           f"/data/datasets/beir/msmarco/XC/{meta_type}/substring_lbl_X_Y.npz",
                           f"/data/datasets/beir/msmarco/XC/trn_X_Y.npz") 

    with open("/home/aiscuser/metrics/msmarco.json", "w") as file:
        json.dump(metrics, file, indent=4)


def linker_inference_helper(pred_dir:str, meta_type:str, meta_name:str, batch_size:Optional[int]=1000):
    metrics = dict() 

    key = os.path.basename(pred_dir.rstrip("/"))
    m = metrics.setdefault(key, {})

    # Test dataset
    m["test"] = infer(f"{pred_dir}/predictions/test_predictions.npz", 
                      f"/data/datasets/beir/msmarco/XC/{meta_type}/{meta_name}_lbl_X_Y.npz",
                      f"/data/datasets/beir/msmarco/XC/tst_X_Y.npz") 

    m["all_test"] = infer(f"{pred_dir}/predictions/test_predictions_all-{meta_name}.npz", 
                          f"/data/datasets/beir/msmarco/XC/{meta_type}/all-{meta_name}_lbl_X_Y.npz",
                          f"/data/datasets/beir/msmarco/XC/tst_X_Y.npz") 

    # Test dataset
    m["train"] = infer(f"{pred_dir}/predictions/train_predictions.npz", 
                       f"/data/datasets/beir/msmarco/XC/{meta_type}/{meta_name}_lbl_X_Y.npz",
                       f"/data/datasets/beir/msmarco/XC/trn_X_Y.npz") 

    return metrics


def linker_inference(batch_size:Optional[int]=1000):
    meta_info = [
        [
            "00_msmarco-gpt-concept-substring-linker-with-ngame-loss-001", 
            "substring", 
            "substring"
        ],
        [
            "06_msmarco-gpt-narrow-substring-linker-with-ngame-loss-001", 
            "narrow_substring", 
            "substring"
        ],
        [
            "07_msmarco-gpt-intent-substring-linker-with-ngame-loss-001", 
            "intent_substring", 
            "intent"
        ],
    ]

    all_metrics = {"msmarco": {}}
    pred_dir = "/data/outputs/upma/"

    for pred_dir_name, meta_type, meta_name in tqdm(meta_info):
        metrics = linker_inference_helper(f"{pred_dir}/{pred_dir_name}", meta_type, meta_name, batch_size=1000)
        all_metrics["msmarco"].update(metrics)

    with open("/home/aiscuser/metrics/linker.json", "w") as file:
        json.dump(all_metrics, file, indent=4)


if __name__ == "__main__":
    linker_inference(batch_size=1000)

