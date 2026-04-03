import scipy.sparse as sp, os, json, numpy as np

from tqdm.auto import tqdm

from xcai.core import *
from xcai.misc import BEIR_DATASETS

from xclib.utils.sparse import retain_topk
import xclib.evaluation.xc_metrics as xm

from scipy.stats import ttest_rel


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
    output_dir = {
        "memory": "/home/sasokan/suchith/outputs/upma/20_upma-ngame-gpt-intent-substring-linker-with-tied-meta-encoder-for-msmarco-002/",
        "no_memory": "/home/sasokan/suchith/outputs/upma/20_upma-ngame-gpt-intent-substring-linker-with-tied-meta-encoder-for-msmarco-003/",
    }

    mem_scores, nme_scores = [], []
    for dataset in tqdm(BEIR_DATASETS):

        # load ground-truth 

        config_file = f"/data/datasets/beir/{dataset}/XC/configs/data.json"
        config_key, fname = get_config_key(config_file)
        with open(config_file) as file:
            config = json.load(file)[config_key]

        tst_lbl = sp.load_npz(config["path"]["test"]["data_lbl"]) 

        # load prediction

        dataset = dataset.replace("/", "-")
        mem_lbl = retain_topk(sp.load_npz(output_dir["memory"] + f"/predictions/test_predictions_{dataset}.npz"), k=10)
        nme_lbl = retain_topk(sp.load_npz(output_dir["no_memory"] + f"/predictions/test_predictions_{dataset}.npz"), k=10)

        # compute metric

        mem_scores.append(ndcg(mem_lbl, tst_lbl, k=10)[:, -1])
        nme_scores.append(ndcg(nme_lbl, tst_lbl, k=10)[:, -1])

    mem_scores = np.hstack(mem_scores)
    nme_scores = np.hstack(nme_scores)

    t_stat, p_value = ttest_rel(mem_scores, nme_scores)
    print(p_value)


