import os, torch,json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp, argparse
from typing import Union, List, Optional
from tqdm.auto import tqdm

from xcai.core import *
from xcai.basics import *
from xcai.analysis import *
from xcai.sdata import SXCDataset
from xcai.data import MainXCDataset, XCDataset
from xcai.sdata import SMetaXCDataset, SXCDataset

from xclib.utils.sparse import retain_topk

def get_dset_for_display(dset:Union[XCDataset,SXCDataset], meta_tuples:Optional[List]=[]):
    kwargs = {k: getattr(dset.data, k) for k in [o for o in vars(dset.data).keys() if not o.startswith('__')]}
    data = type(dset.data)(**kwargs)

    meta_kwargs = dict()
    for meta_name, meta, meta_info in meta_tuples:
        kwargs = {'prefix':meta_name, 'data_meta': meta, 'meta_info': meta_info, 'return_scores': True}
        meta_dset = SMetaXCDataset(**kwargs) if isinstance(dset, SXCDataset) else MetaXCDataset(**kwargs)
        meta_kwargs[f'{meta_name}_meta'] = meta_dset

    return type(dset)(data, **meta_kwargs)

def display(block:TextDataset, output_dir:str, metric_mats:List, for_train:bool, use_all:bool, num_examples:int,
            index_types:Optional[List]=["random", "good", "bad"]):

    assert metric_mats[0].shape == metric_mats[1].shape
    assert metric_mats[0].shape[0] == block.dset.n_data

    example_dir = f"{output_dir}/examples"
    os.makedirs(example_dir, exist_ok=True)

    for index_type in tqdm(index_types, total=len(index_types)):
        if index_type == "random":
            np.random.seed(1000)
            idxs = np.random.permutation(block.dset.n_data)[:num_examples]
        elif index_type == "good":
            scores = pointwise_eval(metric_mats[0], metric_mats[1], topk=5, metric="P")
            scores = np.array(scores.sum(axis=1)).flatten()
            idxs = np.argsort(scores)[:-num_examples:-1]
        elif index_type == "bad":
            scores = pointwise_eval(metric_mats[0], metric_mats[1], topk=5, metric="P")
            scores = np.array(scores.sum(axis=1)).flatten()
            idxs = np.argsort(scores)[:num_examples]

        if for_train:
            fname = f"{example_dir}/train_examples_{index_type}.json"
        else:
            fname = (
                f"{example_dir}/test_examples_all-substrings_{index_type}.json"
                if use_all else
                f"{example_dir}/test_examples_{index_type}.json"
            )
        block.dump(fname, idxs)

def additional_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--num_examples', type=int, default=20)
    parser.add_argument('--index_type', type=str, default='random')
    parser.add_argument('--for_train', action='store_true')
    parser.add_argument('--use_all', action='store_true')
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    # Inputs arguements
    input_args = parse_args()
    extra_args = additional_args()

    input_args.text_mode = True
    input_args.use_sxc_sampler = True
    input_args.pickle_dir = "/home/aiscuser/scratch1/datasets/processed/"

    output_dir = "/data/outputs/upma/00_msmarco-gpt-concept-substring-linker-with-ngame-loss-001"

    assert not extra_args.use_all or not extra_args.for_train, f"All substrings should be used in inference mode."

    # Load basic dataset
    config_file = "/data/datasets/beir/msmarco/XC/configs/data.json"
    config_key, fname = get_config_key(config_file)
    pkl_file = get_pkl_file(input_args.pickle_dir, f"msmarco_{fname}_distilbert-base-uncased", input_args.use_sxc_sampler,
                            input_args.exact, input_args.only_test)
    block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block,
                        only_test=input_args.only_test, main_oversample=True, return_scores=True, n_slbl_samples=1)

    # Load metadata
    meta_tuples = list()

    ## Predicted substrings
    if extra_args.for_train:
        meta_file = f"{output_dir}/predictions/train_predictions.npz"
        info_file = "/data/datasets/beir/msmarco/XC/concept_substrings/raw_data/concept-substring.raw.csv"
    else:
        meta_file = (
            f"{output_dir}/predictions/test_predictions_all-substrings.npz"
            if extra_args.use_all else
            f"{output_dir}/predictions/test_predictions.npz"
        )
        info_file = (
            "/data/datasets/beir/msmarco/XC/concept_substrings/raw_data/all-concept-substring.raw.csv"
            if extra_args.use_all else
            "/data/datasets/beir/msmarco/XC/concept_substrings/raw_data/concept-substring.raw.csv"
        )
    meta_info = Info.from_txt(info_file, info_column_names=["identifier", "input_text"])
    meta_lbl = retain_topk(sp.load_npz(meta_file), k=extra_args.topk)
    meta_tuples.append(["pred-sub", meta_lbl, meta_info])

    ## Ground truth substrings
    if extra_args.for_train:
        meta_file = "/data/datasets/beir/msmarco/XC/concept_substrings/concept-substring_trn_X_Y.npz"
        info_file = "/data/datasets/beir/msmarco/XC/concept_substrings/raw_data/concept-substring.raw.csv"
    else:
        meta_file = "/data/datasets/beir/msmarco/XC/concept_substrings/all-concept-substring_tst_X_Y.npz"
        info_file = "/data/datasets/beir/msmarco/XC/concept_substrings/raw_data/all-concept-substring.raw.csv"
    meta_info = Info.from_txt(info_file, info_column_names=["identifier", "input_text"])
    meta_lbl = sp.load_npz(meta_file)
    meta_tuples.append(["gt-sub", meta_lbl, meta_info])

    ## Predicted categories
    meta_dir = "/data/outputs/mogicX/47_msmarco-gpt-category-linker-007/predictions/"
    meta_file = f"{meta_dir}/train_predictions.npz" if extra_args.for_train else f"{meta_dir}/test_predictions.npz"
    info_file = "/data/datasets/beir/msmarco/XC/concept_substrings/raw_data/concept-substring.raw.csv"

    meta_info = Info.from_txt(info_file, info_column_names=["identifier", "input_text"])
    meta_lbl = sp.load_npz(meta_file)
    meta_tuples.append(["pred-cat", meta_lbl, meta_info])

    ## Prediction block
    pred_block = get_dset_for_display(block.test.dset, meta_tuples)

    # Display predictions
    meta_file = "/data/datasets/beir/msmarco/XC/concept_substrings/concept-substring_tst_X_Y.npz"
    metric_mats = [meta_tuples[0][1], sp.load_npz(meta_file) if not extra_args.for_train and extra_args.use_all else meta_tuples[1][1]]

    disp_block = TextDataset(pred_block, pattern=".*(_text|_scores)$", combine_info=True, sort_by="scores")
    display(disp_block, output_dir, metric_mats, extra_args.for_train, extra_args.use_all, extra_args.num_examples)


