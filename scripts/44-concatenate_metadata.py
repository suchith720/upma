import scipy.sparse as sp, numpy as np, os, json
from typing import List, Optional
from tqdm.auto import tqdm

from sugar.core import load_raw_file, save_raw_file

from xcai.misc import BEIR_DATASETS
from xclib.utils.sparse import retain_topk


def early_concate_metadata(data_dir:str, output_dir:str, dset_type:str, dset_name:str, meta_order:Optional[str]="sorted", 
                           meta_name:Optional[str]="fact", data_type:Optional[str]="test"):

    data_file = f"{data_dir}/{dset_type}/{dset_name}/XC/raw_data/{data_type}.raw.csv"
    data_ids, data_txt = load_raw_file(data_file)

    meta_file = f"{data_dir}/{dset_type}/{dset_name}/XC/raw_data/{meta_name}.raw.csv"
    if not os.path.exists(meta_file): return
    meta_ids, meta_txt = load_raw_file(meta_file)

    dset_tag = dset_name.replace('/', '-')
    dm_file = f"{output_dir}/cross_predictions/{meta_name}/test_predictions_{dset_tag}.npz"
    data_meta = retain_topk(sp.load_npz(dm_file), k=5)

    assert len(data_ids) == data_meta.shape[0]
    assert len(meta_ids) == data_meta.shape[1]

    # Augment metadata to the raw file

    aug_txt = []
    for q,r in zip(data_txt, data_meta):

        if meta_order == "sorted":
            idx = np.argsort(r.data)[::-1]
        elif meta_order == "random":
            idx = np.random.permutation(len(r.data))
        else:
            raise ValueError(f"Invalid order type: {meta_order}.")

        indices = r.indices[idx]
        txt = q + " <CATEGORIES> " + " || ".join([meta_txt[i] for i in indices])
        aug_txt.append(txt)

    save_dir = f"{output_dir}/cross_raw_data/{meta_name}/"
    os.makedirs(save_dir, exist_ok=True)

    if meta_order == "sorted":
        raw_file = f"{save_dir}/{data_type}_{meta_name}_topk-sorted_{dset_tag}.raw.txt"
        exp_file = f"{save_dir}/examples_{meta_name}_topk-sorted_{dset_tag}.json"
    elif meta_order == "random":
        raw_file = f"{save_dir}/{data_type}_{meta_name}_topk-random_{dset_tag}.raw.txt"
        exp_file = f"{save_dir}/examples_{meta_name}_topk-random_{dset_tag}.json"

    save_raw_file(raw_file, data_ids, aug_txt)

    # Save examples

    np.random.seed(100)
    rnd_idx = np.random.permutation(len(data_txt))[:10]

    examples = []
    for idx in rnd_idx:
        sort_idx = np.argsort(data_meta[idx].data)[::-1]
        indices, scores = data_meta[idx].indices[sort_idx], data_meta[idx].data[sort_idx]
        example = {
            "query": data_txt[idx], 
            meta_name: [(meta_txt[i], float(s)) for i,s in zip(indices, scores)],
        }
        examples.append(example)

    with open(exp_file, "w") as file:
        json.dump(examples, file, indent=4)


if __name__ == "__main__":
    data_dir, dset_type = "/data/datasets/", "beir"
    output_dir = "/home/sasokan/suchith/outputs/upma/20_upma-ngame-gpt-intent-substring-linker-with-tied-meta-encoder-for-msmarco-003/"

    for dset_name in tqdm(BEIR_DATASETS):
        early_concate_metadata(data_dir, output_dir, dset_type, dset_name, meta_order="sorted", 
                               meta_name="hipporag-fact", data_type="test")


