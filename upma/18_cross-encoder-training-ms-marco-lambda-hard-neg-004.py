import joblib, os, logging, traceback, torch, scipy.sparse as sp, pandas as pd, json, numpy as np, torch.nn as nn

from datetime import datetime
from tqdm.auto import tqdm
from typing import Optional
from datasets import Dataset, concatenate_datasets, load_dataset

from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator 
from sentence_transformers.cross_encoder.losses import LambdaLoss, NDCGLoss2PPScheme
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.util import mine_hard_negatives

from transformers import AutoTokenizer


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


def _retain_topk(x, k=5):
    data, indices, indptr = [], [], [0]
    for i,j in tqdm(zip(x.indptr, x.indptr[1:]), total=x.shape[0]):
        a,b = x.data[i:j], x.indices[i:j]
        idxs = np.argsort(a)[:-k-1:-1]

        data.extend(a[idxs].tolist())
        indices.extend(b[idxs].tolist())
        indptr.append(len(data))

    mat = sp.csr_matrix((data, indices, indptr), dtype=x.dtype, shape=x.shape)
    mat.sort_indices()
    mat.sum_duplicates()

    return mat


def _get_metadata(indices, meta_info, tokz, num_toks=100):
    n_toks, idx = 0, 0
    meta_items = []
    while n_toks < num_toks:
        text = str(np.random.choice(meta_info[indices[idx]]))
        n_toks += len(tokz.encode(text)[1:-1])
        meta_items.append(text)
        idx = (idx + 1)%len(indices)
    return " [SEP] ".join(meta_items)


def _get_dataset(config, lbl_name, use_meta:Optional[bool]=False, meta_tok_budget:Optional[int]=100, lbl_topk:Optional[int]=None):
    data_info, lbl_info = read_raw_file(config["data_info"]), read_raw_file(config[f"{lbl_name}_info"])

    data_lbl = sp.load_npz(config[f"data_{lbl_name}"])
    data_lbl = data_lbl if lbl_topk is None else _retain_topk(data_lbl, k=lbl_topk)

    relevance = [1] if lbl_name == "lbl" else [0]

    if use_meta:
        tokz = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
        meta_info = [o.split(" [SEP] ") for o in read_raw_file(config["lnk_meta"]["meta_info"])]
        data_meta = _retain_topk(sp.load_npz(config["lnk_meta"]["data_meta"]), k=5)

    queries, docs, labels = [], [], []
    for idx in tqdm(range(data_lbl.shape[0])):
        indices = data_lbl[idx].indices

        queries.append(data_info[idx]) 
        if use_meta: 
            meta_text = _get_metadata(data_meta[idx].indices, meta_info, tokz, meta_tok_budget)
            queries[-1] = queries[-1] + " [SEP] " + meta_text

        docs.append([lbl_info[i] for i in indices])
        labels.append(relevance * len(indices))

    return Dataset.from_dict({"query": queries, "positive": docs, "labels": labels})


def get_positive_dataset(config, use_meta:Optional[bool]=False, meta_tok_budget:Optional[int]=100):
    return _get_dataset(config, "lbl", use_meta=use_meta, meta_tok_budget=meta_tok_budget)

def get_negative_dataset(config, use_meta:Optional[bool]=False, meta_tok_budget:Optional[int]=100, neg_k:Optional[int]=None):
    return _get_dataset(config, "neg", use_meta=use_meta, meta_tok_budget=meta_tok_budget, lbl_topk=neg_k)


def get_eval_dataset(pos_dataset, pred_file, lbl_file, topk:Optional[int]=None):

    pred_lbl = sp.load_npz(pred_file)
    pred_lbl = pred_lbl if topk is None else _retain_topk(pred_lbl, k=topk)

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


def _concatenate_datasets(pos_dataset, neg_dataset):
    assert len(pos_dataset) == len(neg_dataset)

    dataset = []
    for i in tqdm(range(len(pos_dataset))):
        a,b = pos_dataset[i], neg_dataset[i]
        assert a["query"].split(" [SEP] ")[0] == b["query"].split(" [SEP] ")[0]
        o = {
            "query": a["query"],
            "positive": a["positive"] + b["positive"],
            "labels": a["labels"] + b["labels"],
        }
        dataset.append(o)

    return Dataset.from_list(dataset)
        

def main():
    output_dir = "/home/aiscuser/scratch1/outputs/upma/18_cross-encoder-training-ms-marco-lambda-hard-neg-004"
    pickle_dir = "/home/aiscuser/scratch1/datasets/processed/"

    model_name = "microsoft/MiniLM-L12-H384-uncased"

    # Set the log level to INFO to get more information
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # train_batch_size and eval_batch_size inform the size of the batches, while mini_batch_size is used by the loss
    # to subdivide the batch into smaller parts. This mini_batch_size largely informs the training speed and memory usage.
    # Keep in mind that the loss does not process `train_batch_size` pairs, but `train_batch_size * num_docs` pairs.
    train_batch_size = 8
    eval_batch_size = 8
    mini_batch_size = 8
    num_epochs = 1
    max_docs = None
    num_negatives = None

    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 1. Define our CrossEncoder model
    # Set the seed so the new classifier weights are identical in subsequent runs
    torch.manual_seed(12)
    model = CrossEncoder(model_name, num_labels=1)
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/microsoft/ms_marco
    config_key = "data_lbl_ngame-gpt-intent-substring-conflation-01_ce-negatives-topk-05-linker-label-intent-substring_exact"
    config_file = f"configs/msmarco/intent_substring/{config_key}.json"
    with open(config_file) as file:
        config = json.load(file)[config_key]["path"]

    pickle_file = f"{pickle_dir}/{os.path.basename(output_dir)}.joblib"
    if os.path.exists(pickle_file):
        train_dataset, test_dataset, eval_dataset = joblib.load(pickle_file)
    else:
        pos_dataset = get_positive_dataset(config["train"], use_meta=True, meta_tok_budget=50)
        logging.info(f"Created {len(pos_dataset):_} query-positive pairs")

        neg_dataset = get_negative_dataset(config["train"], use_meta=False, neg_k=num_negatives, meta_tok_budget=50)
        logging.info(f"Created {len(neg_dataset):_} query-negative pairs")

        # Concatenate the two datasets into one to  form training dataset
        train_dataset = _concatenate_datasets(pos_dataset, neg_dataset)

        # Evaluation dataset
        pred_file = "/data/outputs/upma/09_upma-with-ngame-gpt-intent-substring-linker-for-msmarco-008/predictions/test_predictions_msmarco.npz"
        lbl_file = "/data/datasets/beir/msmarco/XC/raw_data/label.raw.csv"

        test_dataset = get_positive_dataset(config["test"], use_meta=True)

        eval_dataset = get_eval_dataset(test_dataset, pred_file, lbl_file)

        joblib.dump((train_dataset, test_dataset, eval_dataset), pickle_file)

    logging.info(train_dataset)


    # 3. Define our training loss
    loss = LambdaLoss(
        model=model,
        weighting_scheme=NDCGLoss2PPScheme(),
        mini_batch_size=mini_batch_size,
    )

    # 4. Define the evaluator. We use the CENanoBEIREvaluator, which is a light-weight evaluator for English reranking
    evaluator = CrossEncoderRerankingEvaluator(
        samples=eval_dataset,
        name="ms-marco-dev",
        show_progress_bar=True,
        always_rerank_positives=False,
    )
    # results = evaluator(model)

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-msmarco-{short_model_name}-lambdaloss-hard-neg"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"{output_dir}/{run_name}_{dt}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        load_best_model_at_end=True,
        metric_for_best_model="ms-marco-dev_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=250,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, useful to include these in the model card
    evaluator(model)

    # 8. Save the final model
    final_output_dir = f"{output_dir}/{run_name}_{dt}/final"
    model.save_pretrained(final_output_dir)

if __name__ == "__main__":
    main()

