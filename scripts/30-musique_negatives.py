import scipy.sparse as sp, numpy as np, json

from sugar.core import *

from xclib.utils.sparse import retain_topk

if __name__ == "__main__":

    # Compute negatives

    pred_1 = sp.load_npz("/data/outputs/upma/21_ngame-for-musique-001/predictions/train_predictions.npz")
    pred_2 = sp.load_npz("/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/multihop/musique/train_labels.npz")

    trn_lbl = sp.load_npz("/data/datasets/multihop/musique/XC/trn_X_Y.npz")
    rows, cols = trn_lbl.nonzero()

    pred_1[rows, cols] = 0.0
    pred_1.eliminate_zeros()

    pred_2[rows, cols] = 0.0
    pred_2.eliminate_zeros()

    pred_1 = retain_topk(pred_1, k=25)
    pred_2 = retain_topk(pred_2, k=25)

    neg_lbl = pred_1 + pred_2
    neg_lbl.data[:] = 1.0

    sp.save_npz("/data/datasets/multihop/musique/XC/negatives_trn_X_Y.npz", neg_lbl)

    # Negative examples

    trn_ids, trn_txt = load_raw_file("/data/datasets/multihop/musique/XC/raw_data/train.raw.csv")
    lbl_ids, lbl_txt = load_raw_file("/data/datasets/multihop/musique/XC/raw_data/label.raw.csv")

    neg_lbl = retain_topk(neg_lbl, k=5)

    examples = []
    np.random.seed(100)
    for i in np.random.permutation(len(trn_ids))[:10]:
        example = {
            "query": trn_txt[i],
            "labels": [lbl_txt[o] for o in trn_lbl[i].indices],
            "negatives": [lbl_txt[o] for o in neg_lbl[i].indices],
        }
        examples.append(example)

    with open("/home/sasokan/suchith/outputs/examples/11-musique_negatives.json", "w") as file:
        json.dump(examples, file, indent=4)


