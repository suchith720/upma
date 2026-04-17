import os, pandas as pd, scipy.sparse as sp, json, numpy as np
from tqdm.auto import tqdm

from xclib.utils.sparse import retain_topk

from sugar.core import *

if __name__ == "__main__":

    model_type = "cross_encoder"

    # Load training data

    qry_file = "/data/datasets/beir/msmarco/XC/raw_data/train.raw.csv"
    qry_ids, qry_txt = load_raw_file(qry_file)

    lbl_file = "/data/datasets/beir/msmarco/XC/raw_data/label_exact.raw.txt"
    lbl_ids, lbl_txt = load_raw_file(lbl_file)

    qry_lbl_file = "/data/datasets/beir/msmarco/XC/trn_X_Y_ce-exact.npz"
    qry_lbl = sp.load_npz(qry_lbl_file)


    if model_type == "nvembedv2":

        qry_lbl_file = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/trn_X_Y_normalize-exact.npz"
        nv_qry_lbl = sp.load_npz(qry_lbl_file)

        # Load positives

        pos_file = "/data/datasets/beir/msmarco/XC/raw_data/label.raw.csv"
        pos_ids, pos_txt = load_raw_file(pos_file)

        def load_positives(fname, k=5):
            qry_pos = - sp.load_npz(fname)
            qry_pos = retain_topk(qry_pos, k=k)
            return qry_pos

        qry_pos_file = "/data/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/msmarco/positives_trn_X_Y_normalize_thresh-95.npz"
        qry_pos_1 = load_positives(qry_pos_file, k=5) 

        qry_pos_file = "/data/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/msmarco/positives_trn_X_Y_normalize_thresh-90.npz"
        qry_pos_2 = load_positives(qry_pos_file, k=5) 

        qry_pos_file = "/data/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/msmarco/positives_trn_X_Y_normalize_thresh-85.npz"
        qry_pos_3 = load_positives(qry_pos_file, k=5) 

        qry_pos_file = "/data/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/msmarco/positives_trn_X_Y_normalize_thresh-80.npz"
        qry_pos_4 = load_positives(qry_pos_file, k=5) 

        np.random.seed(1000)

        def sorted_idx(mat, i):
            sort_idx = np.argsort(mat[i].data)[::-1]
            return mat[i].indices[sort_idx], mat[i].data[sort_idx]

        examples = []
        for i in tqdm(np.random.permutation(len(qry_ids))[:1000]):
            indices_1, scores_1 = sorted_idx(qry_pos_1, i)
            indices_2, scores_2 = sorted_idx(qry_pos_2, i)
            indices_3, scores_3 = sorted_idx(qry_pos_3, i)
            indices_4, scores_4 = sorted_idx(qry_pos_4, i)

            examples.append({
                "index": int(i),
                "query": qry_txt[i],
                "labels": [(lbl_txt[p], {"ce": float(q), "nv-embed": float(r)}) for p,q,r in zip(qry_lbl[i].indices, qry_lbl[i].data, nv_qry_lbl[i].data)],

                "thresh 95 positives": [(pos_txt[p], float(q)) for p,q in zip(indices_1, scores_1)],
                "thresh 90 positives": [(pos_txt[p], float(q)) for p,q in zip(indices_2, scores_2)],
                "thresh 85 positives": [(pos_txt[p], float(q)) for p,q in zip(indices_3, scores_3)],
                "thresh 80 positives": [(pos_txt[p], float(q)) for p,q in zip(indices_4, scores_4)],
            })

        fname = "/home/sasokan/suchith/outputs/examples/07-msmarco_positives.json"
        with open(fname, "w") as file:
            json.dump(examples, file, indent=4)


    elif model_type == "cross_encoder":

        # Load positives

        pos_file = "/data/datasets/beir/msmarco/XC/raw_data/ce-scores.raw.txt" 
        pos_ids, pos_txt = load_raw_file(pos_file)

        def load_positives(fname, k=5):
            qry_pos = - sp.load_npz(fname)
            qry_pos = retain_topk(qry_pos, k=k)
            return qry_pos

        qry_pos_file = "/data/datasets/beir/msmarco/XC/cross_encoder/ce-positives-topk-05_trn_X_Y_thresh-95.npz" 
        qry_pos_1 = load_positives(qry_pos_file, k=5) 

        qry_pos_file = "/data/datasets/beir/msmarco/XC/cross_encoder/ce-positives-topk-05_trn_X_Y_thresh-90.npz" 
        qry_pos_2 = load_positives(qry_pos_file, k=5) 

        qry_pos_file = "/data/datasets/beir/msmarco/XC/cross_encoder/ce-positives-topk-05_trn_X_Y_thresh-85.npz" 
        qry_pos_3 = load_positives(qry_pos_file, k=5) 

        np.random.seed(1000)

        def sorted_idx(mat, i):
            sort_idx = np.argsort(mat[i].data)[::-1]
            return mat[i].indices[sort_idx], mat[i].data[sort_idx]

        examples = []
        for i in tqdm(np.random.permutation(len(qry_ids))[:1000]):
            indices_1, scores_1 = sorted_idx(qry_pos_1, i)
            indices_2, scores_2 = sorted_idx(qry_pos_2, i)
            indices_3, scores_3 = sorted_idx(qry_pos_3, i)

            examples.append({
                "index": int(i),
                "query": qry_txt[i],
                "labels": [(lbl_txt[p], float(q)) for p,q in zip(qry_lbl[i].indices, qry_lbl[i].data)],

                "thresh 95 positives": [(pos_txt[p], float(q)) for p,q in zip(indices_1, scores_1)],
                "thresh 90 positives": [(pos_txt[p], float(q)) for p,q in zip(indices_2, scores_2)],
                "thresh 85 positives": [(pos_txt[p], float(q)) for p,q in zip(indices_3, scores_3)],
            })

        fname = "/home/sasokan/suchith/outputs/examples/09-msmarco_ce-positives.json"
        with open(fname, "w") as file:
            json.dump(examples, file, indent=4)
