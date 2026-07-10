import math, pandas as pd, csv, numpy as np, scipy.sparse as sp, os

from tqdm.auto import tqdm
from datasets import load_dataset

from sugar.core import *


def convert_dict2matrix(mapping):
    vocab = dict()
    data, indices, indptr = [], [], [0]
    for k,v in mapping.items():
        for l in v:
            idx = vocab.setdefault(l, len(vocab))
            indices.append(idx)
            data.append(1.0)
        indptr.append(len(indices))
    return vocab, sp.csr_matrix((data, indices, indptr), dtype=np.float32)


def save_datadict(data_dir, dset_name, qry2lbl, lbl_name="label"):
    qry_ids = [f"{dset_name}-query-{i}" for i in range(len(qry2lbl))]
    qry_txt = list(qry2lbl.keys())
    save_raw_file(f"{data_dir}/{dset_name}/raw_data/train.raw.csv", qry_ids, qry_txt)

    lbl_dict, qry_lbl = convert_dict2matrix(qry2lbl)
    qry_lbl_file = f"{data_dir}/{dset_name}/trn_X_Y.npz" if lbl_name == "label" else f"{data_dir}/{dset_name}/{lbl_name}_trn_X_Y.npz"
    sp.save_npz(qry_lbl_file, qry_lbl)

    lbl_ids = [f"{dset_name}-{lbl_name}-{i}" for i in range(len(lbl_dict))]
    lbl_txt = [l for l in sorted(lbl_dict, key=lambda x: lbl_dict[x])]
    save_raw_file(f"{data_dir}/{dset_name}/raw_data/{lbl_name}.raw.csv", lbl_ids, lbl_txt)


if __name__ == "__main__":

    dataset_id = "nomic-ai/nomic-embed-supervised-data"
    dataset = load_dataset(dataset_id)

    data_dir = "/data/datasets/nomic/"

    for dset_name in tqdm(dataset.keys()):

        os.makedirs(f"{data_dir}/{dset_name}/", exist_ok=True)
        os.makedirs(f"{data_dir}/{dset_name}/raw_data", exist_ok=True)

        qry2lbl, qry2neg = dict(), dict()
        for data in tqdm(dataset[dset_name]):
            assert isinstance(data["query"], str)
            assert isinstance(data["document"], str)
            assert isinstance(data["negative"], list)

            qry2lbl.setdefault(data["query"], set()).add(data["document"])
            qry2neg.setdefault(data["query"], set()).update(data["negative"])

        save_datadict(data_dir, dset_name, qry2lbl)
        save_datadict(data_dir, dset_name, qry2neg, lbl_name="negative")



