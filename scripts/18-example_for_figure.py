import scipy.sparse as sp, numpy as np, pandas as pd, json

from sugar.core import *

if __name__ == "__main__":
    trn_lbl = sp.load_npz("/data/datasets/beir/msmarco/XC/trn_X_Y_exact.npz")
    _, trn_raw = load_raw_file("/data/datasets/beir/msmarco/XC/raw_data/train.raw.csv")
    _, lbl_raw = load_raw_file("/data/datasets/beir/msmarco/XC/raw_data/label_exact.raw.txt")

    assert trn_lbl.shape[0] == len(trn_raw)
    assert trn_lbl.shape[1] == len(lbl_raw)

    trn_meta = sp.load_npz("/data/datasets/beir/msmarco/XC/intent_substring/intent_trn_X_Y.npz")
    _, meta_raw = load_raw_file("/data/datasets/beir/msmarco/XC/intent_substring/raw_data/intent.raw.csv")

    assert trn_meta.shape[1] == len(meta_raw)
    assert trn_meta.shape[0] == len(trn_raw)

    trn_conflate = sp.load_npz("/data/datasets/beir/msmarco/XC/intent_substring/conflation_01/intent_trn_X_Y.npz")
    _, conflate_raw = load_raw_file("/data/datasets/beir/msmarco/XC/intent_substring/conflation_01/raw_data/intent.raw.csv")

    trn_pconflate = sp.load_npz("/data/outputs/upma/07_msmarco-gpt-intent-substring-linker-with-ngame-loss-002/predictions/train_predictions.npz")

    assert trn_conflate.shape[1] == len(conflate_raw)
    assert trn_conflate.shape[0] == len(trn_raw)

    examples = []
    conflate_trn = trn_conflate.transpose().tocsr()

    def get_info(idx):
        o = {
            "Query": trn_raw[idx],
            "Labels": [lbl_raw[i] for i in trn_lbl[idx].indices],
            "Intents": [meta_raw[i] for i in trn_meta[idx].indices],
        }
        return o

    np.random.seed(1000)
    for idx in np.random.permutation(trn_lbl.shape[0])[:20]:
        o = get_info(idx)
        o.update({"Conflated": [conflate_raw[i] for i in trn_conflate[idx].indices]})
        o.update({"Predicted metadata": [conflate_raw[i] for i in trn_pconflate[idx].indices]})

        related = []
        for i in trn_conflate[idx].indices:
            for j in conflate_trn[i].indices[:3]:
                related.append(get_info(j))
        o.update({"Related queries": related})
        examples.append(o)

    with open("/home/aiscuser/scratch1/examples/01-figure_examples.json", "w") as file:
        json.dump(examples, file, indent=4)

