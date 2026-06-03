import os, numpy as np, scipy.sparse as sp, pytrec_eval, json

from sugar.core import *
from xcai.misc import *

from typing import Optional, List

def beir_metric(
    inp:sp.csr_matrix,
    targ:sp.csr_matrix,
    qry_ids:Optional[List]=None,
    lbl_ids:Optional[List]=None,
):
    if qry_ids is None: qry_ids = np.arange(inp.shape[0])
    if lbl_ids is None: lbl_ids = np.arange(inp.shape[1])

    assert len(qry_ids) == inp.shape[0], "Query identifiers should be same as number of input queries."
    assert len(lbl_ids) == inp.shape[1], "Label identifiers should be same as number of input labels."

    results = {str(i): {str(lbl_ids[p]):float(q) for p,q in zip(r.indices, r.data)} for i,r in zip(qry_ids, inp)}
    qrels = {str(i): {str(lbl_ids[p]):int(q) for p,q in zip(r.indices, r.data)} for i,r in zip(qry_ids, targ)}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    scores = evaluator.evaluate(results)
    return np.array([scores[str(i)]["ndcg_cut_10"] for i in qry_ids])
    
if __name__ == "__main__":

    def get_examples(dataset):
        # file_1 = f"/data/outputs/mogicX/44_distilbert-gpt-category-linker-oracle-for-msmarco-005/predictions/test_predictions_{dataset}.npz"
        file_1 = f"/data/outputs/upma/26_early-fusion-with-hipporag-fact-exact-for-msmarco-001/predictions/test_predictions_{dataset}.npz"
        pred_lbl_1 = sp.load_npz(file_1)

        file_2 = f"/data/outputs/upma/20_upma-ngame-gpt-intent-substring-linker-with-tied-meta-encoder-for-msmarco-003/predictions/test_predictions_{dataset}.npz"
        pred_lbl_2 = sp.load_npz(file_2)

        # tst_ids, tst_txt = load_raw_file(f"/data/datasets/beir/metadata/{dataset}/raw_data/test_gpt-category-linker.raw.csv")
        tst_ids, tst_txt = load_raw_file(f"/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-003/raw_data/beir/{dataset}/test_hipporag-fact_topk-sorted.raw.txt")
        lbl_ids, lbl_txt = load_raw_file(f"/data/datasets/beir/{dataset}/XC/raw_data/label.raw.csv")
        tst_lbl = sp.load_npz(f"/data/datasets/beir/{dataset}/XC/tst_X_Y.npz")

        scores_1 = beir_metric(pred_lbl_1, tst_lbl, tst_ids, lbl_ids)
        scores_2 = beir_metric(pred_lbl_2, tst_lbl, tst_ids, lbl_ids)

        scores = scores_2 - scores_1
        idxs = np.argsort(scores)[::-1]
        scores = scores[idxs]

        num, examples = 10, []
        for i,sc in zip(idxs[:num], scores[:num]):
            examples.append({
                "query": tst_txt[i],
                "labels": [lbl_txt[l] for l in tst_lbl[i].indices],
                "scores": sc,
            })

        return examples

    examples = []
    examples.extend(get_examples("scifact"))

    fname = "/home/sasokan/suchith/outputs/examples/20-hipporag_memory_useful_examples.json"
    with open(fname, "w") as file:
        json.dump(examples, file, indent=4)

