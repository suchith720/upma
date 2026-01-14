import pandas as pd, os, numpy as np, ast, json_repair, json

from tqdm.auto import tqdm
from typing import Dict, List

from xcai.misc import BEIR_DATASETS


HEAD_DATASETS = [
    "msmarco",
    "climate-fever",
    "dbpedia-entity",
    "fever",
    "hotpotqa",
    "nq",
]


TAIL_DATASETS = [
    "arguana",
    "fiqa",
    "nfcorpus",
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
    

def combine_head_df(dirname:str, dtype:str):
    return pd.concat([pd.read_table(f"{dirname}/{fname}") for fname in os.listdir(dirname) if dtype in fname])


def combine_tail_df(dirname:str, dtype:str):
    folders = [o for o in os.listdir(dirname) if dtype in o]
    assert len(folders) == 1, folders

    dirname = f"{dirname}/{folders[0]}"
    return pd.concat([pd.read_table(f"{dirname}/{fname}") for fname in os.listdir(dirname)])


def convert_string_to_object(output:Dict):
    try:
        if output[:9] == "```python": output = output[9:]
        if output[-3:] == "```": output = output[:-3]
        return ast.literal_eval(output)
    except:
        return json_repair.loads(output)


if __name__ == "__main__":

    output_dir = "/data/share/from_manish/tail_datasets/"
    datasets = TAIL_DATASETS
    combine_df = combine_tail_df
    fname = "tail_datasets"
    types = ["simple", "multihop"]

    examples = list()
    for dataset in tqdm(datasets):
        dataset = dataset.replace("/", "-")
        dirname = f"{output_dir}/{dataset}"
        for dtype in types:

            df = combine_df(dirname, dtype)

            idxs = np.random.permutation(df.shape[0])[:10]
            df = df.iloc[idxs]

            for idx, document, substring in zip(df["id"], df["document"], df["raw_model_response"]):
                substring = convert_string_to_object(substring)["queries"]

                o = {
                    "dataset": dataset,
                    "type": dtype,
                    "id": idx,
                    "document": document,
                    "queries": substring,
                }
                examples.append(o)

    with open(f"/home/aiscuser/{fname}.json", "w") as file:
        json.dump(examples, file, indent=4)



