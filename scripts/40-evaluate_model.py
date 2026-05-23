import os, scipy.sparse as sp, torch

from xcai.main import *
from xcai.misc import *
from xcai.metrics import *

from sugar.core import *

if __name__ == "__main__":

    # data_dir = "/data/outputs/upma/17_upma-with-ngame-gpt-intent-substring-linker-for-msmarco-with-calibration-loss-001/predictions/"
    data_dir = "/data/outputs/upma/20_upma-ngame-gpt-intent-substring-linker-with-tied-meta-encoder-for-msmarco-003/predictions/"

    for dataset in ["trec-covid", "webis-touche2020"]:
        print(dataset)
        data_pred = sp.load_npz(f"{data_dir}/test_predictions_{dataset}.npz")
        data_lbl = sp.load_npz(f"/data/datasets/beir/{dataset}/XC/tst_X_Y.npz")

        tst_ids, tst_txt = load_raw_file(f"/data/datasets/beir/{dataset}/XC/raw_data/test.raw.csv")
        lbl_ids, lbl_txt = load_raw_file(f"/data/datasets/beir/{dataset}/XC/raw_data/label.raw.csv")

        metrics = beir_metric(data_pred, data_lbl, tst_ids, lbl_ids)
        print(metrics)

