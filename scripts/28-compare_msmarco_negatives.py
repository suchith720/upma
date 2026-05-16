import os, pandas as pd, scipy.sparse as sp, json, numpy as np
from tqdm.auto import tqdm

from xclib.utils.sparse import retain_topk

from sugar.core import *

if __name__ == "__main__":

    # Load training data

    qry_file = "/data/datasets/beir/msmarco/XC/raw_data/train.raw.csv"
    qry_ids, qry_txt = load_raw_file(qry_file)

    lbl_file = "/data/datasets/beir/msmarco/XC/raw_data/label.raw.txt"
    lbl_ids, lbl_txt = load_raw_file(lbl_file)

    # qry_lbl_file = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/ce-positives_trn_X_Y_top5.npz"
    qry_lbl_file = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/beir/msmarco/positives_trn_X_Y_normalize_thresh-98-top-5.npz"
    qry_lbl = sp.load_npz(qry_lbl_file)

    # Load negatives

    # neg_file = "/data/datasets/beir/msmarco/XC/raw_data/ce-scores.raw.txt"
    neg_file = "/data/datasets/beir/msmarco/XC/raw_data/label.raw.csv"
    neg_ids, neg_txt = load_raw_file(neg_file)

    def load_mat(fname, k=5):
        qry_neg = sp.load_npz(fname)
        qry_neg = retain_topk(qry_neg, k=k)
        return qry_neg

    # qry_neg_file = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/ce-negatives_trn_X_Y_thresh-70.npz"
    qry_neg_file = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/beir/msmarco/negatives_trn_X_Y_normalize_thresh-70.npz"
    qry_neg = load_mat(qry_neg_file, k=5) 

    assert qry_neg.shape[1] == len(neg_txt)
    assert qry_neg.shape[0] == len(qry_txt)
    assert qry_lbl.shape == qry_neg.shape

    np.random.seed(1000)

    def sorted_idx(mat, i):
        sort_idx = np.argsort(mat[i].data)[::-1]
        return mat[i].indices[sort_idx], mat[i].data[sort_idx]

    # examples

    examples = []
    for i in tqdm(np.random.permutation(len(qry_ids))[:20]):
        indices, scores = sorted_idx(qry_neg, i)

        examples.append({
            "index": int(i),
            "query": qry_txt[i],
            "labels": [(lbl_txt[p], float(q)) for p,q in zip(qry_lbl[i].indices, qry_lbl[i].data)],
            "thresh 70 negatives": [(neg_txt[p], float(q)) for p,q in zip(indices, scores)],
        })

        fname = "/home/sasokan/suchith/outputs/examples/13-msmarco_nvembedv2-negatives_thresh-70.json"
        with open(fname, "w") as file:
            json.dump(examples, file, indent=4)

