import joblib, os, logging, traceback, torch, scipy.sparse as sp, pandas as pd, json, numpy as np, torch.nn as nn

from typing import Optional
from tqdm.auto import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset

from sentence_transformers.util import mine_hard_negatives
from sentence_transformers import SentenceTransformer

def load_raw_txt(fname:str, encoding:str='utf-8', sep:Optional[str]='->'):
    ids, raw = [], []
    with open(fname, 'r', encoding=encoding) as file:
        for line in file:
            k, v = line[:-1].split(sep, maxsplit=1)
            ids.append(k); raw.append(v)
    return ids, raw

def load_raw_csv(fname:str, id_name:str="identifier", raw_name:str="text"):
    df = pd.read_csv(fname)
    df.fillna('', inplace=True)
    ids, raw = df[id_name].tolist(), df[raw_name].tolist()
    return ids, raw

def load_raw_file(fname:str, id_name:Optional[str]="identifier", raw_name:Optional[str]="text", 
                  sep:Optional[str]='->', encoding:Optional[str]='utf-8'):
    if fname.endswith(".txt"): 
        return load_raw_txt(fname, encoding=encoding, sep=sep)
    elif fname.endswith(".csv"): 
        return load_raw_csv(fname, id_name=id_name, raw_name=raw_name)
    else: 
        raise ValueError(f"Invalid filename: {fname}.")

def read_raw_file(fname):
    return load_raw_file(fname)[1]

def get_positive_dataset(config):
    data_info, lbl_info = read_raw_file(config["data_info"]), read_raw_file(config["lbl_info"])
    data_lbl = sp.load_npz(config["data_lbl"])

    queries, docs, labels = [], [], []
    for idx in tqdm(range(data_lbl.shape[0])):
        indices = data_lbl[idx].indices
        queries.append(data_info[idx]) 
        docs.append([lbl_info[i] for i in indices])
        labels.append([1] * len(indices))

    return Dataset.from_dict({"query": queries, "positive": docs, "labels": labels})

def get_eval_dataset(pos_dataset, pred_file, lbl_file):

    pred_lbl = sp.load_npz(pred_file)
    lbl_info = pd.read_csv(lbl_file)["text"]

    assert len(pos_dataset) == pred_lbl.shape[0]

    eval_dataset = []
    for idx in range(pred_lbl.shape[0]):
        o = pos_dataset[idx]

        indices = pred_lbl[idx].indices
        sort_idx = np.argsort(pred_lbl[idx].data)[::-1]
        documents = [lbl_info[i] for i in indices[sort_idx]]
        o.update({"documents": documents})

        eval_dataset.append(o)

    return eval_dataset

def get_mined_hard_negatives(pos_dataset, lbl_file):
    # 2b. Prepare the hard negative dataset by mining hard negatives
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding_model_batch_size = 1024
    skip_n_hardest = 3
    num_hard_negatives = 9  # 1 positive + 9 negatives
    
    all_passages = read_raw_file(lbl_file)
    logging.info(f"Corpus contains {len(all_passages):_} unique passages")

    queries, positives = [], []
    for item in pos_dataset:
        for passage in item["positive"]:
            queries.append(item["query"])
            positives.append(passage)
    pairs_dataset = Dataset.from_dict({"query": queries, "positive": positives})
    logging.info(f"Created {len(pairs_dataset):_} query-positive pairs")

    hard_negatives_dataset = mine_hard_negatives(
        dataset=pairs_dataset,
        model=embedding_model,
        corpus=all_passages,  # Use all passages as the corpus
        num_negatives=num_hard_negatives,
        range_min=skip_n_hardest,  # Skip the most similar passages
        range_max=skip_n_hardest + num_hard_negatives * 3,  # Look for negatives in a reasonable range
        batch_size=embedding_model_batch_size,
        output_format="labeled-list",
        use_faiss=True,
    )

    return hard_negatives_dataset

if __name__ == "__main__":
    pickle_dir = "/home/aiscuser/scratch1/datasets/processed/"

    config_key = "data_lbl_ngame-gpt-intent-substring-conflation-01_ce-negatives-topk-05-linker-label-intent-substring_exact"
    config_file = f"configs/msmarco/intent_substring/{config_key}.json"
    with open(config_file) as file:
        config = json.load(file)[config_key]["path"]

    pickle_file = f"{pickle_dir}/cross-encoder-training-ms-marco-lambda-hard-neg.joblib"
    if os.path.exists(pickle_file):
        train_dataset, test_dataset, eval_dataset = joblib.load(pickle_file)
    else:
        lbl_file = "/data/datasets/beir/msmarco/XC/raw_data/label.raw.csv"

        pos_dataset = get_positive_dataset(config["train"])
        logging.info(f"Created {len(pos_dataset):_} query-positive pairs")

        neg_dataset = get_mined_hard_negatives(pos_dataset, lbl_file)
        logging.info(f"Created {len(neg_dataset):_} query-negative pairs")

        # Evaluation dataset
        pred_file = "/data/outputs/upma/09_upma-with-ngame-gpt-intent-substring-linker-for-msmarco-008/predictions/test_predictions_msmarco.npz"

        test_dataset = get_positive_dataset(config["test"])

        eval_dataset = get_eval_dataset(test_dataset, pred_file, lbl_file)

        joblib.dump((pos_dataset, neg_dataset, test_dataset, eval_dataset), pickle_file)

