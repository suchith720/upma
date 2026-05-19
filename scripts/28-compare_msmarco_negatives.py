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
    neg_dir = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/beir/msmarco/"
    qry_neg_files = [
        f"{neg_dir}/negatives_trn_X_Y_normalize_thresh-70.npz",
        f"{neg_dir}/negatives_trn_X_Y_normalize_thresh-80.npz",
        f"{neg_dir}/negatives_trn_X_Y_normalize_thresh-85.npz",
    ]
    qry_negs = [load_mat(f, k=5) for f in qry_neg_files]

    for qry_neg in qry_negs:
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

        examples.append({
            "index": int(i),
            "query": qry_txt[i],
            "labels": [(lbl_txt[p], float(q)) for p,q in zip(qry_lbl[i].indices, qry_lbl[i].data)],
        })

        for t, qry_neg in zip([70, 80, 85], qry_negs):
            indices, scores = sorted_idx(qry_neg, i)
            examples[-1].update(
                {
                    f"thresh {t} negatives": [(neg_txt[p], float(q)) for p,q in zip(indices, scores)],
            })

        fname = "/home/sasokan/suchith/outputs/examples/15-msmarco_nvembedv2-negatives_thresh-70-80-85.json"
        with open(fname, "w") as file:
            json.dump(examples, file, indent=4)

