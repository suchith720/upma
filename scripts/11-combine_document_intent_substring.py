import pandas as pd, os, numpy as np, scipy.sparse as sp
from tqdm.auto import tqdm

from xcai.misc import BEIR_DATASETS


def load_info(data_dir:str):
    lbl_intent = sp.load_npz(f"{data_dir}/document_intent_substring/simple/intent_lbl_X_Y.npz")
    intent_info = pd.read_csv(f"{data_dir}/document_intent_substring/simple/raw_data/label_intent.raw.csv")
    lbl_info = pd.read_csv(f"{data_dir}/raw_data/label.raw.csv")

    lbl_idxs = np.where(lbl_intent.getnnz(axis=1) > 0)[0]
    intent_idxs = np.where(lbl_intent.getnnz(axis=0) > 0)[0]

    lbl_intent = lbl_intent[lbl_idxs][:, intent_idxs]
    lbl_intent.sort_indices()
    lbl_intent.sum_duplicates()

    lbl_info = lbl_info.iloc[lbl_idxs]
    intent_info = intent_info.iloc[intent_idxs]
        
    assert lbl_intent.shape[0] == lbl_info.shape[0]
    assert lbl_intent.shape[1] == intent_info.shape[0]

    return lbl_intent, lbl_info, intent_info


if __name__ == "__main__":

    all_lbl_info, all_intent_info  = [], []

    all_data, all_indices, all_indptr = [], [], [0]

    n_intent = 0
    for dataset in tqdm(BEIR_DATASETS):
        if dataset == "quora": continue

        data_dir = f"/data/datasets/beir/{dataset}/XC/"
        lbl_intent, lbl_info, intent_info = load_info(data_dir)

        all_lbl_info.append(lbl_info)
        all_intent_info.append(intent_info)

        all_data.extend(lbl_intent.data.tolist())

        indices = lbl_intent.indices + n_intent
        all_indices.extend(indices.tolist()) 

        indptr = lbl_intent.indptr + all_indptr[-1] 
        all_indptr.extend(indptr[1:].tolist())

        n_intent += lbl_intent.shape[1]

    all_lbl_intent = sp.csr_matrix((all_data, all_indices, all_indptr), dtype=np.float32)

    all_lbl_info = pd.concat(all_lbl_info, axis=0)
    all_intent_info = pd.concat(all_intent_info, axis=0)

    save_dir = "/data/datasets/beir/experiments/00_beir-gpt-document-intent-substring/"
    os.makedirs(f"{save_dir}/raw_data", exist_ok=True)

    sp.save_npz(f"{save_dir}/intent_lbl_X_Y.npz", all_lbl_intent)
    all_lbl_info.to_csv(f"{save_dir}/raw_data/label.raw.csv", index=False)
    all_intent_info.to_csv(f"{save_dir}/raw_data/intent.raw.csv", index=False)




        
