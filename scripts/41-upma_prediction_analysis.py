import torch, scipy.sparse as sp, numpy as np, pytrec_eval, json

from sugar.core import *
from xclib.utils.sparse import retain_topk


def get_qrels(tst_lbl, tst_ids, lbl_ids):
    qrels = dict()
    for i, r in zip(tst_ids, tst_lbl):
        qrels[i] = {lbl_ids[idx]:int(sc) for idx, sc in zip(r.indices, r.data)}
    return qrels

def get_run(pred_lbl, tst_ids, lbl_ids):
    run = dict()
    for i, r in zip(tst_ids, pred_lbl):
        run[i] = {lbl_ids[idx]:float(sc) for idx, sc in zip(r.indices, r.data)}
    return run

if __name__ == "__main__":
    dataset = "arguana"
    # dataset = "dbpedia-entity"

    tst_ids, tst_txt = load_raw_file(f"/data/datasets/beir/{dataset}/XC/raw_data/test.raw.csv")
    lbl_ids, lbl_txt = load_raw_file(f"/data/datasets/beir/{dataset}/XC/raw_data/label.raw.csv")

    fname = f"/data/outputs/upma/20_upma-ngame-gpt-intent-substring-linker-with-tied-meta-encoder-for-msmarco-003/predictions/test_predictions_{dataset}.npz"
    pred_lbl_1 = sp.load_npz(fname)

    fname = f"/data/outputs/upma/17_upma-with-ngame-gpt-intent-substring-linker-for-msmarco-with-calibration-loss-001/predictions/test_predictions_{dataset}.npz"
    pred_lbl_2 = sp.load_npz(fname)

    assert pred_lbl_1.shape == pred_lbl_2.shape

    data_meta = sp.load_npz(f"/data/outputs/upma/07_msmarco-gpt-intent-substring-linker-with-ngame-loss-002/predictions/test_predictions_{dataset}.npz")
    data_meta = retain_topk(data_meta, k=5)
    meta_ids, meta_txt = load_raw_file("/data/datasets/beir/msmarco/XC/intent_substring/conflation_01/raw_data/intent.raw.csv")

    assert data_meta.shape[0] == pred_lbl_1.shape[0]

    tst_lbl = sp.load_npz(f"/data/datasets/beir/{dataset}/XC/tst_X_Y.npz")
    qrels = get_qrels(tst_lbl, tst_ids, lbl_ids)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})

    run_1 = get_run(pred_lbl_1, tst_ids, lbl_ids)
    run_2 = get_run(pred_lbl_2, tst_ids, lbl_ids)

    results = evaluator.evaluate(run_1)
    metric_1 = np.array([results[o]["ndcg_cut_10"] for o in tst_ids])

    results = evaluator.evaluate(run_2)
    metric_2 = np.array([results[o]["ndcg_cut_10"] for o in tst_ids])

    metric_diff = metric_2 - metric_1
    sort_idxs = np.argsort(metric_diff)[::-1]
    sort_idxs = sort_idxs[:10]

    examples = []
    for idx, sc in zip(sort_idxs, metric_diff[sort_idxs]):
        examples.append({
            "query": tst_txt[idx],
            "labels": [lbl_txt[i] for i in tst_lbl[idx].indices],
            "metadata": [meta_txt[i] for i in data_meta[idx].indices],
            "score": f"{sc:.4f}",
        })

    exp_file = f"/home/sasokan/suchith/outputs/examples/17-memtr_{dataset}_examples.json"
    with open(exp_file, "w") as file:
        json.dump(examples, file, indent=4)

