import json, numpy as np, scipy.sparse as sp

from sugar.core import load_raw_file


def display_data(data_dir, dset_name, exp_file):
    np.random.seed(100)
    data_ids, data_txt = load_raw_file(f"{data_dir}/{dset_name}/raw_data/train.raw.csv")

    lbl_ids, lbl_txt = load_raw_file(f"{data_dir}/{dset_name}/raw_data/label.raw.csv")
    data_lbl = sp.load_npz(f"{data_dir}/{dset_name}/trn_X_Y.npz")

    neg_ids, neg_txt = load_raw_file(f"{data_dir}/{dset_name}/raw_data/negative.raw.csv")
    data_neg = sp.load_npz(f"{data_dir}/{dset_name}/negative_trn_X_Y.npz")

    rnd_idx = np.random.permutation(len(data_txt))[:10]

    examples = []
    for idx in rnd_idx:
        lbl_idxs, neg_idxs = data_lbl[idx].indices, data_neg[idx].indices
        example = {
            "query": data_txt[idx], 
            "labels": [lbl_txt[i] for i in lbl_idxs],
            "negatives": [neg_txt[i] for i in neg_idxs],
        }
        examples.append(example)

    with open(exp_file, "w") as file:
        json.dump(examples, file, indent=4)

if __name__ == "__main__":
    data_dir = "/data/datasets/nomic/"
    dset_name = "msmarco_distillation_simlm_rescored_reranked_min15"

    exp_file = f"/data/suchith/datasets/examples/01-{dset_name}.json"

    display_data(data_dir, dset_name, exp_file)
