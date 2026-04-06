import os, pandas as pd, scipy.sparse as sp, json, numpy as np
from tqdm.auto import tqdm

from xclib.utils.sparse import retain_topk

from sugar.core import *

if __name__ == "__main__":
    qry_file = "/data/datasets/beir/msmarco/XC/raw_data/train.raw.csv"
    qry_ids, qry_txt = load_raw_file(qry_file)

    lbl_file = "/data/datasets/beir/msmarco/XC/raw_data/label_exact.raw.txt"
    lbl_ids, lbl_txt = load_raw_file(lbl_file)

    neg_file = "/data/datasets/beir/msmarco/XC/raw_data/ce-scores.raw.txt"
    neg_ids, neg_txt = load_raw_file(neg_file)

    qry_neg_file = "/data/datasets/beir/msmarco/XC/ce-negatives-topk-05_trn_X_Y.npz" 
    qry_neg = sp.load_npz(qry_neg_file)

    qry_lbl_file = "/data/datasets/beir/msmarco/XC/trn_X_Y_ce-exact.npz"
    qry_lbl = sp.load_npz(qry_lbl_file)

    qry_neg = retain_topk(qry_neg, k=5)

    np.random.seed(1000)

    examples = []
    for i in tqdm(np.random.permutation(len(qry_ids))[:20]):
        sort_idx = np.argsort(qry_neg[i].data)[::-1]

        indices = qry_neg[i].indices[sort_idx]
        scores = qry_neg[i].data[sort_idx]

        examples.append({
            "index": int(i),
            "query": qry_txt[i],
            "labels": [(lbl_txt[p], float(q)) for p,q in zip(qry_lbl[i].indices, qry_lbl[i].data)],
            "negatives": [(neg_txt[p], float(q)) for p,q in zip(indices, scores)],
        })

    fname = "/home/sasokan/suchith/outputs/examples/04-msmarco_negatives.json"
    with open(fname, "w") as file:
        json.dump(examples, file, indent=4)

