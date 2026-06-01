import os, scipy.sparse as sp, torch, json

from tqdm.auto import tqdm

from xcai.main import *
from xcai.misc import *
from xcai.metrics import *

from sugar.core import *

if __name__ == "__main__":

    # expt_name = "17_upma-with-ngame-gpt-intent-substring-linker-for-msmarco-with-calibration-loss-001"
    # data_dir = f"/data/outputs/upma/{expt_name}/cross_predictions/07_msmarco-gpt-intent-substring-linker-with-ngame-loss-002/from_document-intent-substring-simple-label-intent_to_intent/using_16_beir-gpt-intent-substring-query-linker-with-ngame-loss-001/"
    # meta_file = f"/data/outputs/upma/{expt_name}/cross_metrics/07_msmarco-gpt-intent-substring-linker-with-ngame-loss-002/from_document-intent-substring-simple-label-intent_to_intent/using_16_beir-gpt-intent-substring-query-linker-with-ngame-loss-001/beir.json"

    # expt_name = "20_upma-ngame-gpt-intent-substring-linker-with-tied-meta-encoder-for-msmarco-003"
    # data_dir = f"/data/outputs/upma/{expt_name}/predictions/"
    # meta_file = f"/data/outputs/upma/{expt_name}/metrics/beir.json"

    # expt_name = "17_upma-with-ngame-gpt-intent-substring-linker-for-msmarco-with-calibration-loss-001"
    # data_dir = f"/data/outputs/upma/{expt_name}/cross_predictions/16_beir-gpt-intent-substring-query-linker-with-ngame-loss-001/intent-conflation-01/"
    # meta_file = f"/data/outputs/upma/{expt_name}/cross_metrics/16_beir-gpt-intent-substring-query-linker-with-ngame-loss-001/intent-conflation-01/beir.json"

    # data_dir = "/data/outputs/mogicX/37_training-msmarco-distilbert-from-scratch-008/predictions/"
    # meta_file = "/data/outputs/mogicX/37_training-msmarco-distilbert-from-scratch-008/metrics/beir.json"

    # data_dir = "/data/suchith/outputs/upma/25_early-fusion-with-category-metadata-gpt5-linker-for-msmarco-001/predictions/"
    # meta_file = "/data/suchith/outputs/upma/25_early-fusion-with-category-metadata-gpt5-linker-for-msmarco-001/metrics/beir.json"

    data_dir = "/data/suchith/outputs/upma/25_early-fusion-with-category-metadata-gpt5-linker-for-msmarco-002/predictions/"
    meta_file = "/data/suchith/outputs/upma/25_early-fusion-with-category-metadata-gpt5-linker-for-msmarco-002/metrics/beir.json"

    metrics = dict()

    for dataset in tqdm(BEIR_DATASETS):
        dset_tag = dataset.replace("/", "-")

        pred_file = f"{data_dir}/test_predictions_{dset_tag}.npz"
        if not os.path.exists(pred_file): continue

        data_pred = sp.load_npz(pred_file)
        data_lbl = sp.load_npz(f"/data/datasets/beir/{dataset}/XC/tst_X_Y.npz")

        tst_ids, tst_txt = load_raw_file(f"/data/datasets/beir/{dataset}/XC/raw_data/test.raw.csv")
        lbl_ids, lbl_txt = load_raw_file(f"/data/datasets/beir/{dataset}/XC/raw_data/label.raw.csv")

        m = beir_metric(data_pred, data_lbl, tst_ids, lbl_ids)
        metrics[dataset] = m

    with open(meta_file, "w") as file:
        json.dump(metrics, file, indent=4)

