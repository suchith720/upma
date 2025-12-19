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

DATASETS = [
    "msmarco",
    "arguana",
    "climate-fever",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "webis-touche2020",
    "trec-covid",
    "cqadupstack/android",
    "cqadupstack/english",
    "cqadupstack/gaming",
    "cqadupstack/gis",
    "cqadupstack/mathematica",
    "cqadupstack/physics",
    "cqadupstack/programmers",
    "cqadupstack/stats",
    "cqadupstack/tex",
    "cqadupstack/unix",
    "cqadupstack/webmasters",
    "cqadupstack/wordpress"
]


def get_pred_dset_for_display(pred:sp.csr_matrix, dset:Union[XCDataset,SXCDataset], meta_tuples:Optional[List]=[]):
    kwargs = {k: getattr(dset.data, k) for k in [o for o in vars(dset.data).keys() if not o.startswith('__')]}
    data = type(dset.data)(**kwargs)
    
    kwargs = {'prefix':'pred', 'data_meta': pred, 'meta_info': dset.data.lbl_info, 'return_scores': True}
    pred_dset = SMetaXCDataset(**kwargs) if isinstance(dset, SXCDataset) else MetaXCDataset(**kwargs)

    meta_kwargs = dict()
    for meta_name, meta, meta_info in meta_tuples:
        kwargs = {'prefix':meta_name, 'data_meta': meta, 'meta_info': meta_info, 'return_scores': True}
        meta_dset = SMetaXCDataset(**kwargs) if isinstance(dset, SXCDataset) else MetaXCDataset(**kwargs)
        meta_kwargs[f'{meta_name}_meta'] = meta_dset

    meta_kwargs['pred_meta'] = pred_dset
    
    return type(dset)(data, **meta_kwargs)


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

    ## dataset

    input_args.text_mode = True
    input_args.use_sxc_sampler = True
    input_args.pickle_dir = "/home/aiscuser/scratch1/datasets/processed/"

    assert not extra_args.use_all or not extra_args.for_train, f"All substrings should be used in inference mode."

    if extra_args.use_all:
        config_file = "/data/datasets/beir/msmarco/XC/configs/data_gpt-all-concept-substring.json"
    else:
        config_file = "/data/datasets/beir/msmarco/XC/configs/data_gpt-concept-substring.json"
    config_key, fname = get_config_key(config_file)

    pkl_file = get_pkl_file(input_args.pickle_dir, f"msmarco_{fname}_distilbert-base-uncased", input_args.use_sxc_sampler, 
                            input_args.exact, input_args.only_test)

    ## predictions

    output_dir = "/home/aiscuser/scratch1/outputs/upma/00_msmarco-gpt-concept-substring-linker-with-ngame-loss-001"

    # Load data

    ## Load block
    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, only_test=input_args.only_test, 
                        n_slbl_samples=1, main_oversample=False)
    dset = block.train.dset if extra_args.for_train else block.test.dset

    ## Load substring metadata
    if extra_args.for_train:
        pred_file = f"{output_dir}/predictions/train_predictions.npz" 
    else:
        pred_file = (
            f"{output_dir}/predictions/test_predictions_all-substrings.npz"
            if extra_args.use_all else 
            f"{output_dir}/predictions/test_predictions.npz" 
        )
    pred_lbl = retain_topk(sp.load_npz(pred_file), k=extra_args.topk)

    ## Load category metadata
    meta_dir = "/data/outputs/mogicX/47_msmarco-gpt-category-linker-007/"
    meta_file = f"{meta_dir}/predictions/train_predictions.npz" if extra_args.for_train else f"{meta_dir}/predictions/test_predictions.npz"
    data_meta = retain_topk(sp.load_npz(meta_file), k=extra_args.topk)

    info_file = "/data/datasets/beir/msmarco/XC/raw_data/category-gpt-linker_conflated-001_conflated-001.raw.csv" 
    meta_info = Info.from_txt(info_file, info_column_names=["identifier", "input_text"])

    pred_block = get_pred_dset_for_display(pred_lbl, dset=dset, meta_tuples=[("cat", data_meta, meta_info)])

    # Display predictions

    disp_block = TextDataset(pred_block, pattern='.*(_text|_scores)$', combine_info=True, sort_by='scores')
    
    example_dir = f"{output_dir}/examples"
    os.makedirs(example_dir, exist_ok=True)

    for index_type in tqdm(["random", "good", "bad"], total=3):
        if index_type == "random":
            np.random.seed(1000)
            idxs = np.random.permutation(block.test.dset.n_data)[:extra_args.num_examples]
        elif index_type == "good":
            scores = pointwise_eval(pred_lbl, dset.data.data_lbl, topk=5, metric="P")
            scores = np.array(scores.sum(axis=1)).flatten()
            idxs = np.argsort(scores)[:-extra_args.num_examples:-1]
        elif index_type == "bad":
            scores = pointwise_eval(pred_lbl, dset.data.data_lbl, topk=5, metric="P")
            scores = np.array(scores.sum(axis=1)).flatten()
            idxs = np.argsort(scores)[:extra_args.num_examples]

        if extra_args.for_train:
            fname = f"{example_dir}/train_examples_{index_type}.json" 
        else:
            fname = (
                f"{example_dir}/test_examples_all-substrings_{index_type}.json"
                if extra_args.use_all else 
                f"{example_dir}/test_examples_{index_type}.json" 
            )

        disp_block.dump(fname, idxs)

