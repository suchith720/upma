import os, pandas as pd, scipy.sparse as sp, json, numpy as np
from tqdm.auto import tqdm

from xclib.utils.sparse import retain_topk

from sugar.core import *

if __name__ == "__main__":

    # Load training data

    qry_file = "/data/datasets/beir/msmarco/XC/raw_data/train.raw.csv"
    qry_ids, qry_txt = load_raw_file(qry_file)

    lbl_file = "/data/datasets/beir/msmarco/XC/raw_data/label_exact.raw.txt"
    lbl_ids, lbl_txt = load_raw_file(lbl_file)

    qry_lbl_file = "/data/datasets/beir/msmarco/XC/trn_X_Y_ce-exact.npz"
    qry_lbl = sp.load_npz(qry_lbl_file)

    qry_lbl_file = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/trn_X_Y_normalize-exact.npz"
    nv_qry_lbl = sp.load_npz(qry_lbl_file)

    # Load negatives

    neg_file = "/data/datasets/beir/msmarco/XC/raw_data/ce-scores.raw.txt"
    neg_ids, neg_txt = load_raw_file(neg_file)

    def load_negatives(fname, k=5):
        qry_neg = sp.load_npz(fname)
        qry_neg = retain_topk(qry_neg, k=k)
        return qry_neg

    qry_neg_file = "/data/datasets/beir/msmarco/XC/ce-negatives-topk-05_trn_X_Y.npz" 
    qry_neg = load_negatives(qry_neg_file, k=5) 

    # Load thresholded negatives

    qry_neg_file = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/negatives_trn_X_Y_normalize_thresh-50.npz" 
    qry_th_neg_1 = load_negatives(qry_neg_file, k=5) 

    qry_neg_file = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/negatives_trn_X_Y_normalize_thresh-70.npz" 
    qry_th_neg_2 = load_negatives(qry_neg_file, k=5) 

    qry_neg_file = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/negatives_trn_X_Y_normalize_thresh-90.npz" 
    qry_th_neg_3 = load_negatives(qry_neg_file, k=5) 

    qry_neg_file = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/negatives_trn_X_Y_normalize_thresh-95.npz" 
    qry_th_neg_4 = load_negatives(qry_neg_file, k=5) 

    np.random.seed(1000)

    def sorted_idx(mat):
        sort_idx = np.argsort(mat[i].data)[::-1]
        return mat[i].indices[sort_idx], mat[i].data[sort_idx]

    examples = []
    for i in tqdm(np.random.permutation(len(qry_ids))[:10_000]):
        indices_1, scores_1 = sorted_idx(qry_neg)
        indices_2, scores_2 = sorted_idx(qry_th_neg_1)
        indices_3, scores_3 = sorted_idx(qry_th_neg_2)
        indices_4, scores_4 = sorted_idx(qry_th_neg_3)
        indices_5, scores_5 = sorted_idx(qry_th_neg_4)

        examples.append({
            "index": int(i),
            "query": qry_txt[i],
            "labels": [(lbl_txt[p], {"ce": float(q), "nv-embed": float(r)}) for p,q,r in zip(qry_lbl[i].indices, qry_lbl[i].data, nv_qry_lbl[i].data)],

            "original negatives": [(neg_txt[p], float(q)) for p,q in zip(indices_1, scores_1)],

            "thresh 50 negatives": [(neg_txt[p], float(q)) for p,q in zip(indices_2, scores_2)],
            "thresh 70 negatives": [(neg_txt[p], float(q)) for p,q in zip(indices_3, scores_3)],
            "thresh 90 negatives": [(neg_txt[p], float(q)) for p,q in zip(indices_4, scores_4)],
            "thresh 95 negatives": [(neg_txt[p], float(q)) for p,q in zip(indices_5, scores_5)],
        })

    fname = "/home/sasokan/suchith/outputs/examples/04-msmarco_negatives.json"
    with open(fname, "w") as file:
        json.dump(examples, file, indent=4)

