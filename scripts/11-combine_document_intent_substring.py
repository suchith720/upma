import pandas as pd, os, numpy as np, scipy.sparse as sp
from tqdm.auto import tqdm

from xcai.misc import BEIR_DATASETS


def load_info(mat_file:str, data_file:str, intent_file:str):
    data_intent = sp.load_npz(mat_file)

    data_info = pd.read_csv(data_file)
    intent_info = pd.read_csv(intent_file)

    data_idxs = np.where(data_intent.getnnz(axis=1) > 0)[0]
    intent_idxs = np.where(data_intent.getnnz(axis=0) > 0)[0]

    data_intent = data_intent[data_idxs][:, intent_idxs]
    data_intent.sort_indices()
    data_intent.sum_duplicates()

    data_info = data_info.iloc[data_idxs]
    intent_info = intent_info.iloc[intent_idxs]
        
    assert data_intent.shape[0] == data_info.shape[0]
    assert data_intent.shape[1] == intent_info.shape[0]

    return data_intent, data_info, intent_info


def get_beir_data_metadata(data_meta_file:str, data_file:str, meta_file:str, save_dir:str, data_name:str, meta_name:str):
    all_data_info, all_meta_info  = [], []

    all_data, all_indices, all_indptr = [], [], [0]

    n_meta = 0
    for dataset in tqdm(BEIR_DATASETS):
        if dataset == "quora": continue

        lbl_intent_file = f"{data_dir}/document_intent_substring/simple/intent_lbl_X_Y.npz"

        lbl_info_file = f"{data_dir}/raw_data/label.raw.csv"
        intent_info_file = f"{data_dir}/document_intent_substring/simple/raw_data/label_intent.raw.csv"

        data_meta, data_info, meta_info = load_info(data_meta_file.format(dataset=dataset), 
                                                    data_file.format(dataset=dataset), 
                                                    meta_file.format(dataset=dataset))

        all_data_info.append(data_info)
        all_meta_info.append(meta_info)

        all_data.extend(data_meta.data.tolist())

        indices = data_meta.indices + n_meta
        all_indices.extend(indices.tolist()) 

        indptr = data_meta.indptr + all_indptr[-1] 
        all_indptr.extend(indptr[1:].tolist())

        n_meta += data_meta.shape[1]

    all_data_meta = sp.csr_matrix((all_data, all_indices, all_indptr), dtype=np.float32)

    all_data_info = pd.concat(all_data_info, axis=0)
    all_meta_info = pd.concat(all_meta_info, axis=0)

    os.makedirs(f"{save_dir}/raw_data", exist_ok=True)

    sp.save_npz(f"{save_dir}/{meta_name}_{data_name}.npz", all_data_meta)
    all_data_info.to_csv(f"{save_dir}/raw_data/{data_name}.raw.csv", index=False)
    all_meta_info.to_csv(f"{save_dir}/raw_data/{data_name}_{meta_name}.raw.csv", index=False)


if __name__ == "__main__":
    data_dir = "/data/datasets/beir/{dataset}/XC/"

    # data_meta_file = data_dir + "document_intent_substring/simple/intent_lbl_X_Y.npz"
    # data_file = data_dir + "raw_data/label.raw.csv"
    # meta_file = data_dir + "document_intent_substring/simple/raw_data/label_intent.raw.csv"
    # data_name, meta_name = "label", "intent"

    data_meta_file = data_dir + "document_intent_substring/multihop/intent_qry_X_Y.npz"
    data_file = data_dir + "document_intent_substring/multihop/raw_data/query.raw.csv"
    meta_file = data_dir + "document_intent_substring/multihop/raw_data/label_intent.raw.csv"
    data_name, meta_name = "multihop-query", "intent"

    save_dir = "/data/datasets/beir/experiments/00_beir-gpt-document-intent-substring/"
    
    get_beir_data_metadata(data_meta_file, data_file, meta_file, save_dir, data_name, meta_name)

