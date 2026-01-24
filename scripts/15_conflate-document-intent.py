import scipy.sparse as sp, pandas as pd
from tqdm.auto import tqdm

from xcai.misc import BEIR_DATASETS
from xcai.conflate import *

def load_data(data_dir:str):
    qry_file = f"{data_dir}/intent_qry_X_Y.npz"
    lbl_file = f"{data_dir}/intent_lbl_X_Y.npz"
    
    info_file = f"{data_dir}/raw_data/label_intent.raw.csv"
    
    qry_meta = sp.load_npz(qry_file)
    lbl_meta = sp.load_npz(lbl_file)
    
    meta_info = pd.read_csv(info_file)

    return qry_meta, lbl_meta, meta_info, (qry_file, lbl_file, info_file)

if __name__ == "__main__":

    for dataset in tqdm(BEIR_DATASETS):
        data_dir = f"/data/datasets/beir/{dataset}/XC/document_intent_substring/simple/"

        linker_dir = "/data/outputs/upma/12_beir-gpt-intent-substring-document-linker-with-ngame-loss-001/"
        meta_file = f"{linker_dir}/cross_predictions/document-intent-substring_simple/test_predictions_labels_{dataset}.npz"

        save_dir = f"/data/datasets/beir/{dataset}/XC/document_intent_substring/simple/"

        qry_meta, lbl_meta, meta_info, files = load_data(data_dir)
        qry_file, lbl_file, info_file = files

        output = perform_similarity_based_conflation_01(meta_file, qry_meta, None, lbl_meta, meta_info=meta_info, diff_thresh=0.1, sim_topk=1)
        ctrn_meta, cmeta_info, meta_idx, _, clbl_meta = output

        SaveData.proc(save_dir, trn_file=qry_file, trn_meta=ctrn_meta, info_file=info_file, meta_info=cmeta_info, 
                      lbl_file=lbl_file, lbl_meta=clbl_meta)

