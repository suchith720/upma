import torch, scipy.sparse as sp, numpy as np, pandas as pd, json

from xcai.basics import *

import xclib.evaluation.xc_metrics as xm


def _ndcg(eval_flags, n, k=5):
    _cumsum = 0
    _dcg = np.cumsum(np.multiply(
        eval_flags, 1/np.log2(np.arange(k)+2)),
        axis=-1)
    ndcg = []
    for _k in range(k):
        _cumsum += 1/np.log2(_k+1+1)
        ndcg.append(np.multiply(_dcg[:, _k].reshape(-1, 1), 1/np.minimum(n, _cumsum)))
    return np.hstack(ndcg)


def ndcg(X, true_labels, k=5, sorted=False, use_cython=False):
    indices, true_labels, _, _ = xm._setup_metric(
        X, true_labels, k=k, sorted=sorted, use_cython=use_cython)
    eval_flags = xm._eval_flags(indices, true_labels, None)
    _total_pos = np.asarray(
        true_labels.sum(axis=1),
        dtype=np.int32)
    _max_pos = max(np.max(_total_pos), k)
    _cumsum = np.cumsum(1/np.log2(np.arange(1, _max_pos+1)+1))
    n = _cumsum[_total_pos - 1]
    return _ndcg(eval_flags, n, k)


if __name__ == "__main__":

    pickle_dir = "/home/aiscuser/scratch1/datasets/processed/"

    dataset = "scifact"
    config_file = f"/data/datasets/beir/{dataset}/XC/configs/data.json"
    config_key, fname = get_config_key(config_file)

    dataset = dataset.replace("/", "-")
    pkl_file = get_pkl_file(pickle_dir, f"{dataset}_{fname}_distilbert-base-uncased", use_sxc_sampler=True, use_only_test=True)
    block = build_block(pkl_file, config_file, use_sxc=True, config_key=config_key, only_test=True, main_oversample=True,
                        return_scores=True, n_slbl_samples=1)

    files = [
        "/data/outputs/mogicX/37_training-msmarco-distilbert-from-scratch-008/predictions/test_predictions_scifact.npz",
        "/data/outputs/upma/11_early-fusion-with-ngame-gpt-intent-substring-linker-for-msmarco-001/predictions/test_predictions_scifact.npz", 
        "/data/outputs/upma/11_early-fusion-with-ngame-gpt-intent-substring-linker-for-msmarco-001/cross_predictions/document-intent-substring_simple/test_predictions_scifact.npz",
    ]

    data_pred_1 = sp.load_npz(files[0])
    data_pred_2 = sp.load_npz(files[1])
    data_pred_3 = sp.load_npz(files[2])

    def get_scores(data_pred):
        return ndcg(data_pred, block.test.dset.data.data_lbl, k=10)[:, -1]

    scores_1 = get_scores(data_pred_1)
    scores_2 = get_scores(data_pred_2)
    scores_3 = get_scores(data_pred_3)

    idx_1 = np.where(scores_1 < scores_2)[0]
    idx_2 = np.where(scores_2 < scores_3)[0]

    idxs = np.intersect1d(idx_1, idx_2)
    diff_scores = scores_3 - scores_2
    idxs_scores = diff_scores[idxs]

    sort_idxs = np.argsort(idxs_scores)[::-1]

    raw_file_2 = "/data/outputs/upma/07_msmarco-gpt-intent-substring-linker-with-ngame-loss-001/raw_data/test_scifact.raw.csv"
    raw_2 = pd.read_csv(raw_file_2)

    raw_file_3 = "/data/outputs/upma/07_msmarco-gpt-intent-substring-linker-with-ngame-loss-001/cross_raw_data/document-intent-substring_simple/test_scifact.raw.csv"
    raw_3 = pd.read_csv(raw_file_3)

    # Examples

    data_info = block.test.dset.data.data_info["input_text"]
    lbl_info = block.test.dset.data.lbl_info["input_text"]
    data_lbl = block.test.dset.data.data_lbl

    examples = []
    for idx in sort_idxs:
        o = {
            "Query": data_info[idx],
            "Labels": [lbl_info[i] for i in data_lbl[idx].indices],
            "MSMARCO metadata": raw_2["text"].iloc[idx],
            "BeIR metadata": raw_3["text"].iloc[idx],
        }
        examples.append(o)

    fname = "/home/aiscuser/scratch1/examples/01-scifact_task_specific_example.json"
    with open(fname, "w") as file:
        json.dump(examples, file, indent=4)

