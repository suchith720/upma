import scipy.sparse as sp, os, numpy as np
from tqdm.auto import tqdm

from sugar.core import *

if __name__ == "__main__":

    data_dir = "/data/datasets/beir/"
    save_dir = "/data/datasets/beir/all-beir/XC/"
    datasets = ['fever', 'fiqa', 'hotpotqa', 'msmarco', 'nfcorpus', 'nq-train', 'scifact']

    all_trn_ids, all_trn_txt = [], []
    all_lbl_ids, all_lbl_txt = [], []

    num_labels, num_pairs = 0, 0
    data, indices, indptr = [], [], []
    for dset in tqdm(datasets, total=len(datasets)):

        fname = f"{data_dir}/{dset}/XC/raw_data/train.raw.csv"
        trn_ids, trn_txt = load_raw_file(fname)

        fname = f"{data_dir}/{dset}/XC/raw_data/label.raw.csv"
        lbl_ids, lbl_txt = load_raw_file(fname)

        trn_ids = [f"{dset}_{o}" for o in trn_ids]
        lbl_ids = [f"{dset}_{o}" for o in lbl_ids]

        all_trn_ids.extend(trn_ids)
        all_trn_txt.extend(trn_txt)

        all_lbl_ids.extend(lbl_ids)
        all_lbl_txt.extend(lbl_txt)

        trn_lbl = sp.load_npz(f"{data_dir}/{dset}/XC/trn_X_Y.npz")

        assert trn_lbl.shape[0] == len(trn_ids), dset
        assert trn_lbl.shape[1] == len(lbl_ids), dset

        data.extend(trn_lbl.data)
        indices.extend(trn_lbl.indices + num_labels)
        if num_labels == 0:
            indptr.extend(trn_lbl.indptr)
        else:
            indptr.extend(trn_lbl.indptr[1:] + indptr[-1])

        num_labels += trn_lbl.shape[1]
        num_pairs += len(trn_lbl.data)

        assert indptr[-1] == num_pairs, f"{indptr[-1]} != {num_pairs}"
        
    save_raw_file(f"{save_dir}/raw_data/train.raw.csv", all_trn_ids, all_trn_txt)

    mat = sp.csr_matrix((data, indices, indptr), dtype=trn_lbl.dtype, shape=(len(all_trn_ids), len(all_lbl_ids)))
    sp.save_npz(f"{save_dir}/trn_X_Y.npz", mat)

    idxs = np.where(mat.getnnz(axis=0) > 0)[0]

    mat = mat[:, idxs]
    all_lbl_ids = [all_lbl_ids[i] for i in idxs]
    all_lbl_txt = [all_lbl_txt[i] for i in idxs]

    save_raw_file(f"{save_dir}/raw_data/label.raw.csv", all_lbl_ids, all_lbl_txt)

    # Examples

    print("Matrix shape: ", mat.shape)

    print("\nExamples:\n")
    idxs = np.random.permutation(mat.shape[0])[:5]
    for i in idxs:
        print("Query: ", all_trn_txt[i])
        print("Labels: ")
        for j in mat[i].indices:
            print("-- ", all_lbl_txt[j])
        print()

