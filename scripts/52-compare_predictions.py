import torch, scipy.sparse as sp, numpy as np, pytrec_eval, json

from sugar.core import *
from xclib.utils.sparse import retain_topk


def get_qrels(tst_lbl, tst_ids, lbl_ids):
    qrels = dict()
    for i, r in zip(tst_ids, tst_lbl):
        qrels[str(i)] = {str(lbl_ids[idx]):int(sc) for idx, sc in zip(r.indices, r.data)}
    return qrels

def get_run(pred_lbl, tst_ids, lbl_ids):
    run = dict()
    for i, r in zip(tst_ids, pred_lbl):
        run[str(i)] = {str(lbl_ids[idx]):float(sc) for idx, sc in zip(r.indices, r.data)}
    return run

if __name__ == "__main__":

    # dset_num = 4
    # dataset = "scifact"

    # dset_num = 5
    # dataset = "scidocs"

    dset_num = 6
    dataset = "nfcorpus"

    # Ground-truth data

    tst_ids, tst_txt = load_raw_file(f"/data/datasets/beir/{dataset}/XC/raw_data/test.raw.csv")
    lbl_ids, lbl_txt = load_raw_file(f"/data/datasets/beir/{dataset}/XC/raw_data/label.raw.csv")

    tst_lbl = sp.load_npz(f"/data/datasets/beir/{dataset}/XC/tst_X_Y.npz")
    qrels = get_qrels(tst_lbl, tst_ids, lbl_ids)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})

    # Predicted data

    fname = f"/data/outputs/mogicX/44_distilbert-gpt-category-linker-oracle-for-msmarco-005/cross_predictions/gpt-category-linker/test_predictions_{dataset}.npz"
    pred_lbl_1 = sp.load_npz(fname)
    fname = f"/data/outputs/upma/26_early-fusion-with-hipporag-fact-exact-for-msmarco-001/predictions/test_predictions_{dataset}.npz"
    pred_lbl_2 = sp.load_npz(fname)
    assert pred_lbl_1.shape[0] == pred_lbl_2.shape[0]

    # Metadata

    tst_meta_ids_1, tst_meta_txt_1 = load_raw_file(f"/data/datasets/beir/metadata/{dataset}/raw_data/test_gpt-category-linker.raw.csv")
    tst_meta_ids_2, tst_meta_txt_2 = load_raw_file(f"/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-003/raw_data/beir/{dataset}/test_hipporag-fact_topk-sorted.raw.txt")

    run_1 = get_run(pred_lbl_1, tst_ids, lbl_ids)
    run_2 = get_run(pred_lbl_2, tst_ids, lbl_ids)

    results = evaluator.evaluate(run_1)
    metric_1 = np.array([results[str(o)]["ndcg_cut_10"] for o in tst_ids])

    results = evaluator.evaluate(run_2)
    metric_2 = np.array([results[str(o)]["ndcg_cut_10"] for o in tst_ids])

    metric_diff = metric_2 - metric_1
    sort_idxs = np.argsort(metric_diff) # [::-1]
    sort_idxs = sort_idxs[:10]

    # Examples

    examples = []
    pred_lbl_1 = retain_topk(pred_lbl_1, k=5)
    pred_lbl_2 = retain_topk(pred_lbl_2, k=5)

    for idx, sc in zip(sort_idxs, metric_diff[sort_idxs]):
        pred_lbl_1_indices = pred_lbl_1[idx].indices
        sort_idxs = np.argsort(pred_lbl_1[idx].data)[::-1]
        pred_lbl_1_indices = pred_lbl_1_indices[sort_idxs]

        pred_lbl_2_indices = pred_lbl_2[idx].indices
        sort_idxs = np.argsort(pred_lbl_2[idx].data)[::-1]
        pred_lbl_2_indices = pred_lbl_2_indices[sort_idxs]

        examples.append({
            "Category query": tst_meta_txt_1[idx],
            "Hipporag-fact query": tst_meta_txt_2[idx],
            "labels": [lbl_txt[i] for i in tst_lbl[idx].indices],
            "Category labels": [lbl_txt[i] for i in pred_lbl_1_indices],
            "Hipporag-fact labels": [lbl_txt[i] for i in pred_lbl_2_indices],
            "score": f"{sc:.4f}",
        })

    exp_file = f"/data/suchith/outputs/examples/{dset_num:02d}-category_vs_hipporag-fact_{dataset}_examples.json"
    with open(exp_file, "w") as file:
        json.dump(examples, file, indent=4)

