import scipy.sparse as sp, pytrec_eval, numpy as np, torch

from xcai.metrics import *

import xclib.evaluation.xc_metrics as xm


def _setup_metric(X, true_labels, inv_psp=None,
                  k=5, sorted=False, use_cython=False):
    assert xm.compatible_shapes(X, true_labels), \
        "ground truth and prediction matrices must have same shape."
    num_instances, num_labels = true_labels.shape
    indices = xm._get_topk(X, num_labels, k, sorted, use_cython)
    ps_indices = None
    if inv_psp is not None:
        _mat = sp.spdiags(inv_psp, diags=0,
                          m=num_labels, n=num_labels)
        _psp_wtd = xm._broad_cast(_mat.dot(true_labels.T).T, true_labels)
        ps_indices = xm._get_topk(_psp_wtd, num_labels, k, False, use_cython)
        inv_psp = np.hstack([inv_psp, np.zeros((1))])

    idx_dtype = true_labels.indices.dtype
    true_labels = sp.csr_matrix(
        (true_labels.data, true_labels.indices, true_labels.indptr),
        shape=(num_instances, num_labels+1), dtype=true_labels.dtype)

    # scipy won't respect the dtype of indices
    # may fail otherwise on really large datasets
    true_labels.indices = true_labels.indices.astype(idx_dtype)
    return indices, true_labels, ps_indices, inv_psp


def _eval_flags(indices, true_labels, inv_psp=None):
    if sp.issparse(true_labels):
        nr, nc = indices.shape
        rows = np.repeat(np.arange(nr).reshape(-1, 1), nc)
        eval_flags = true_labels[rows, indices.ravel()].A1.reshape(nr, nc)
    elif type(true_labels) == np.ndarray:
        eval_flags = np.take_along_axis(true_labels,
                                        indices, axis=-1)
    if inv_psp is not None:
        eval_flags = np.multiply(inv_psp[indices], eval_flags)
    return eval_flags


def _ndcg(eval_flags, idcg, k=5):
    """
    eval_flags: The relevance scores of the top-k predicted items.
    idcg: The maximum possible DCG for each instance.
    """
    # 1. Calculate Gain: (2^rel - 1)
    # gains = np.power(2, eval_flags) - 1
    gains = eval_flags

    # 2. Calculate Discount: 1/log2(rank + 1)
    discounts = 1 / np.log2(np.arange(k) + 2)

    # 3. Calculate DCG at each k
    # Multiply gains by discounts and take cumulative sum across columns
    dcg = np.cumsum(gains * discounts, axis=-1)

    # 4. Normalize
    # idcg should be (num_instances, k)
    ndcg_per_instance = dcg / idcg

    # Return the mean nDCG@k across all instances
    return np.mean(ndcg_per_instance, axis=0)


def _get_idcg(true_labels, k):
    """
    Calculates the Ideal DCG by sorting all true relevance scores per row.
    """
    num_instances, num_labels = true_labels.shape

    discounts = 1 / np.log2(np.arange(k) + 2)
    discounts = discounts.reshape(1, -1)

    indices = xm._get_topk(true_labels, num_labels, k, True, False)
    idx_dtype = true_labels.indices.dtype
    true_labels = sp.csr_matrix(
        (true_labels.data, true_labels.indices, true_labels.indptr),
        shape=(num_instances, num_labels+1), dtype=true_labels.dtype)
    true_labels.indices = true_labels.indices.astype(idx_dtype)
    scores = _eval_flags(indices, true_labels, None)

    # ideal_gains = np.power(2, scores) - 1
    ideal_gains = scores
    idcg = np.cumsum(ideal_gains * discounts, axis=1)

    assert np.all(idcg != 0)

    return idcg


def ndcg(X, true_labels, k=5, sorted=False, use_cython=False):
    # Setup metrics and get indices of top-k predictions
    indices, true_labels_ext, _, _ = _setup_metric(
        X, true_labels, k=k, sorted=sorted, use_cython=use_cython)

    # Extract the relevance scores for the predicted indices
    # (Uses your existing _eval_flags logic)
    eval_flags = _eval_flags(indices, true_labels_ext, None)

    # Calculate IDCG based on the ground truth relevance scores
    # Note: Use the original true_labels here to get actual scores
    idcg = _get_idcg(true_labels, k)

    return _ndcg(eval_flags, idcg, k)


if __name__ == "__main__":
    # gt = sp.load_npz("/home/sasokan/b-sprabhu/datasets/beir/trecdl19/XC/tst_X_Y.npz")
    # pred = sp.load_npz("/home/sasokan/b-sprabhu/outputs/upma/17_upma-with-ngame-gpt-intent-substring-linker-for-msmarco-with-calibration-loss-001/predictions/test_predictions_trecdl19.npz")

    # gt = sp.load_npz("/home/sasokan/b-sprabhu/datasets/beir/trecdl20/XC/tst_X_Y.npz")
    # pred = sp.load_npz("/home/sasokan/b-sprabhu/outputs/upma/17_upma-with-ngame-gpt-intent-substring-linker-for-msmarco-with-calibration-loss-001/predictions/test_predictions_trecdl20.npz")

    gt = sp.load_npz("/home/sasokan/b-sprabhu/datasets/beir/trecdl19/XC/tst_X_Y.npz")
    pred = sp.load_npz("/data/suchith/outputs/upma/22_upma-with-ngame-gpt-intent-substring-linker-for-msmarco-with-calibration-loss-and-nvembed-teacher-001/predictions/test_predictions_trecdl19.npz")

    # qrels = {}
    # for i in range(gt.shape[0]):
    #     qrels[f"q{i}"] = {f"d{p}": int(q) for p,q in zip(gt[i].indices, gt[i].data)}

    # run = {}
    # for i in range(pred.shape[0]):
    #     run[f"q{i}"] = {f"d{p}": float(q) for p,q in zip(pred[i].indices, pred[i].data)}

    # evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    # results = evaluator.evaluate(run)

    # mean_ndcg = sum(r["ndcg_cut_10"] for r in results.values()) / len(results)
    # print("Mean NDCG@10:", mean_ndcg)

    # m = ndcg(pred, gt, k=10)
    # print("xclib NDCG@10:", m[-1])

    metric = PrecReclMrr(gt.shape[1], pk=10, rk=200, rep_pk=[1, 3, 5, 10], rep_rk=[10, 100, 200], mk=[5, 10, 20])
    o = {
        "pred_idx": torch.tensor(pred.indices),
        "pred_score": torch.tensor(pred.data),
        "pred_ptr": torch.tensor([p-q for p,q in zip(pred.indptr[1:], pred.indptr)]),
        "targ_idx": torch.tensor(gt.indices),
        "targ_score": torch.tensor(gt.data),
        "targ_ptr": torch.tensor([p-q for p,q in zip(gt.indptr[1:], gt.indptr)]),
    }
    print(metric(**o))

