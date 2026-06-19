import os, scipy.sparse as sp, numpy as np
from tqdm.auto import tqdm

from datasets import load_dataset

from sugar.core import *

if __name__ == "__main__":
    dataset = load_dataset("nomic-ai/nomic-embed-supervised-data")

    for k in tqdm(dataset.keys()):

        data_dir = f"/data/datasets/nomic/{k}"
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/raw_data", exist_ok=True)

        qry_lbl_dict, qry_neg_dict = dict(), dict()
        for q,l in tqdm(zip(dataset[k]["query"], dataset[k]["document"]), total=len(dataset[k]["query"])):
            qry_lbl_dict.setdefault(q, []).append(l)

        # query

        qry_txt = list(qry_lbl_dict.keys())
        qry_ids = [f"{k}-query-{i}" for i in range(len(qry_txt))]
        save_raw_file(f"{data_dir}/raw_data/train.raw.csv", qry_ids, qry_txt)

        # matrix

        vocab = dict()
        data, indices, indptr = [], [], [0]
        for q in tqdm(qry_txt):
            for l in qry_lbl_dict[q]:
                idx = vocab.setdefault(l, len(vocab))
                data.append(1.0)
                indices.append(idx)
            indptr.append(len(indices))

        matrix = sp.csr_matrix((data, indices, indptr), dtype=np.float32, shape=(len(qry_txt), len(vocab)))
        sp.save_npz(f"{data_dir}/trn_X_Y.npz", matrix)

        # labels

        lbl_txt = list(sorted(vocab, key=lambda x: vocab[x]))
        lbl_ids = [f"{k}-label-{i}" for i in range(len(lbl_txt))]
        save_raw_file(f"{data_dir}/raw_data/label.raw.csv", qry_ids, qry_txt)

