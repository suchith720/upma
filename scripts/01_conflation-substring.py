from xcai.conflate import *

if __name__ == "__main__":
    data_dir = "/data/datasets/beir/msmarco/XC/substring/"
    meta_type = "substring"

    linker_dir = "/data/outputs/upma/00_msmarco-gpt-concept-substring-linker-with-ngame-loss-001/"
    meta_file = f"{linker_dir}/predictions/test_predictions_labels_msmarco-distilbert-cos-v5.npz"

    save_dir = "/data/datasets/beir/msmarco/XC/substring/"

    (trn_meta, tst_meta, lbl_meta), meta_info, meta_phrases, files = load_data(data_dir, meta_type)
    trn_file, tst_file, lbl_file, info_file = files

    # output = perform_similarity_based_conflation_01(meta_file, trn_meta, tst_meta, lbl_meta, meta_info)
    output = perform_phrase_based_conflation_02(meta_phrases, meta_file, trn_meta, tst_meta, lbl_meta, meta_info)
    ctrn_meta, cmeta_info, meta_idx, ctst_meta, clbl_meta = output

    SaveData.proc(save_dir, trn_file=trn_file, trn_meta=ctrn_meta, info_file=info_file, meta_info=cmeta_info, 
                  tst_file=tst_file, tst_meta=ctst_meta, lbl_file=lbl_file, lbl_meta=clbl_meta)

