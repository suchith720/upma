import scipy.sparse as sp, os, numpy as np
from tqdm.auto import tqdm

from sugar.core import *
from xcai.misc import BEIR_DATASETS

def prepare_labels():
    data_dir = "/data/datasets/beir/sampled-beir/"

    all_lbl_ids, all_lbl_txt = [],  []
    remaining_lbl_ids, remaining_lbl_txt = [], []

    hotpot_lbl_ids, hotpot_lbl_txt = [], []
    msmarco_lbl_ids, msmarco_lbl_txt = [], []

    dataset_stats = dict()

    # head datasets

    for dset in tqdm(os.listdir(f"{data_dir}/head_datasets/")):
        file = f"{data_dir}/head_datasets/{dset}/label_cluster_samples.raw.txt"
        sampled_lbl_ids, sampled_lbl_txt = load_raw_file(file, sep="\t")

        sampled_lbl_set = set(all_lbl_ids)
        lbl_file = f"/data/datasets/beir/{dset}/XC/raw_data/label.raw.csv"
        lbl_ids, lbl_txt = load_raw_file(lbl_file)
        missing_lbl_mask = [l not in sampled_lbl_set for l in lbl_ids]

        missing_lbl_ids = [l for m,l in zip(missing_lbl_mask, lbl_ids) if m]
        missing_lbl_txt = [l for m,l in zip(missing_lbl_mask, lbl_txt) if m]

        sampled_lbl_ids = [f"{dset}_{o}" for o in sampled_lbl_ids]
        all_lbl_ids.extend(sampled_lbl_ids)
        all_lbl_txt.extend(sampled_lbl_txt)

        dataset_stats[dset] = {}
        dataset_stats[dset]["sample"] = len(all_lbl_ids)

        missing_lbl_ids = [f"{dset}_{o}" for o in missing_lbl_ids]
        remaining_lbl_ids.extend(missing_lbl_ids)
        remaining_lbl_txt.extend(missing_lbl_txt)

        dataset_stats[dset]["remaining"] = len(missing_lbl_ids)

        if dset == "hotpotqa":
            hotpot_lbl_ids.extend(missing_lbl_ids)
            hotpot_lbl_txt.extend(missing_lbl_txt)

        elif dset == "msmarco":
            msmarco_lbl_ids.extend(missing_lbl_ids)
            msmarco_lbl_txt.extend(missing_lbl_txt)

    # tail datasets

    for dset in tqdm(os.listdir(f"{data_dir}/tail_datasets/")):
        file = f"{data_dir}/tail_datasets/{dset}/label.raw.txt"
        lbl_ids, lbl_txt = load_raw_file(file, sep="\t")
        lbl_ids = [f"{dset}_{o}" for o in lbl_ids]
        all_lbl_ids.extend(lbl_ids)
        all_lbl_txt.extend(lbl_txt)

        dataset_stats[dset] = {}
        dataset_stats[dset]["sample"] = len(all_lbl_ids)

    print(dataset_stats)

    save_file = f"{data_dir}/beir_label_cluster_samples.raw.txt"
    save_raw_file(save_file, all_lbl_ids, all_lbl_txt, sep="\t")

    save_file = f"{data_dir}/beir_label_remaining_samples.raw.txt"
    save_raw_file(save_file, remaining_lbl_ids, remaining_lbl_txt, sep="\t")

    save_file = f"{data_dir}/hotpotqa_label_remaining_samples.raw.txt"
    save_raw_file(save_file, hotpot_lbl_ids, hotpot_lbl_txt, sep="\t")

    save_file = f"{data_dir}/msmarco_label_remaining_samples.raw.txt"
    save_raw_file(save_file, msmarco_lbl_ids, msmarco_lbl_txt, sep="\t")


def prepare_test_queries():
    data_dir = "/data/datasets/beir/"
    save_dir = "/data/datasets/beir/sampled-beir/"

    all_tst_ids, all_tst_txt = [], []
    sampled_tst_ids, sampled_tst_txt = [], []

    dataset_stats = dict()

    HEAD_DATASETS = set(['climate-fever', 'dbpedia-entity', 'fever', 'hotpotqa', 'msmarco', 'nq', 'trecdl19', 'trecdl20'])
    
    for dset in BEIR_DATASETS:
        if dset == "quora": continue

        fname = f"{data_dir}/{dset}/XC/raw_data/test.raw.csv"
        tst_ids, tst_txt = load_raw_file(fname)

        dset_tag = dset.replace("/", "-")
        dset_type = "head_datasets" if dset_tag in HEAD_DATASETS else "tail_datasets"
        
        fname = f"{save_dir}/{dset_type}"
        fname = (
            f"{fname}/msmarco/{dset}.raw.txt" 
            if dset in set(["trecdl19", "trecdl20"]) else 
            f"{fname}/{dset_tag}/test.raw.txt"
        )
        save_raw_file(fname, tst_ids, tst_txt, sep="\t")

        tst_ids = [f"{dset_tag}_{o}" for o in tst_ids]
        all_tst_ids.extend(tst_ids)
        all_tst_txt.extend(tst_txt)
        
        rnd_idx = np.random.permutation(len(tst_ids))[:2]
        sampled_tst_ids.extend([tst_ids[i] for i in rnd_idx])
        sampled_tst_txt.extend([tst_txt[i] for i in rnd_idx])

    save_file = f"{save_dir}/combined/beir_test.raw.txt"
    save_raw_file(save_file, all_tst_ids, all_tst_txt, sep="\t")

    save_file = f"{save_dir}/combined/beir_test_samples.raw.txt"
    save_raw_file(save_file, sampled_tst_ids, sampled_tst_txt, sep="\t")


def prepare_train_queries():
    data_dir = "/data/datasets/beir/"
    save_dir = "/data/datasets/beir/sampled-beir/"

    all_trn_ids, all_trn_txt = [], []

    HEAD_DATASETS = set(['climate-fever', 'dbpedia-entity', 'fever', 'hotpotqa', 'msmarco', 'nq', 'trecdl19', 'trecdl20'])
    
    for dset in ['fever', 'fiqa', 'hotpotqa', 'msmarco', 'nfcorpus', 'nq-train', 'scifact']:

        fname = f"{data_dir}/{dset}/XC/raw_data/train.raw.csv"
        trn_ids, trn_txt = load_raw_file(fname)

        dset_tag = "nq" if dset == "nq-train" else dset
        dset_type = "head_datasets" if dset_tag in HEAD_DATASETS else "tail_datasets"
        
        fname = f"{save_dir}/{dset_type}"
        fname = f"{fname}/{dset_tag}/train.raw.txt"
        save_raw_file(fname, trn_ids, trn_txt, sep="\t")

        trn_ids = [f"{dset_tag}_{o}" for o in trn_ids]
        all_trn_ids.extend(trn_ids)
        all_trn_txt.extend(trn_txt)
        
    save_file = f"{save_dir}/combined/beir_train.raw.txt"
    save_raw_file(save_file, all_trn_ids, all_trn_txt, sep="\t")


if __name__ == "__main__":
    prepare_labels()


