import scipy.sparse as sp, pytrec_eval

if __name__ == "__main__":
    # gt = sp.load_npz("/home/sasokan/b-sprabhu/datasets/beir/trecdl19/XC/tst_X_Y.npz")
    # pred = sp.load_npz("/home/sasokan/b-sprabhu/outputs/upma/17_upma-with-ngame-gpt-intent-substring-linker-for-msmarco-with-calibration-loss-001/predictions/test_predictions_trecdl19.npz")

    gt = sp.load_npz("/home/sasokan/b-sprabhu/datasets/beir/trecdl20/XC/tst_X_Y.npz")
    pred = sp.load_npz("/home/sasokan/b-sprabhu/outputs/upma/17_upma-with-ngame-gpt-intent-substring-linker-for-msmarco-with-calibration-loss-001/predictions/test_predictions_trecdl20.npz")

    qrels = {}
    for i in range(gt.shape[0]):
        qrels[f"q{i}"] = {f"d{p}": int(q) for p,q in zip(gt[i].indices, gt[i].data)}

    run = {}
    for i in range(pred.shape[0]):
        run[f"q{i}"] = {f"d{p}": float(q) for p,q in zip(pred[i].indices, pred[i].data)}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    results = evaluator.evaluate(run)

    mean_ndcg = sum(r["ndcg_cut_10"] for r in results.values()) / len(results)
    print("Mean NDCG@10:", mean_ndcg)

