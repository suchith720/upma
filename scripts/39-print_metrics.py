import json, argparse, os
from tqdm.auto import tqdm

from xcai.misc import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdir', type=str)
    return parser.parse_known_args()[0]

if __name__ == "__main__":

    input_args = parse_args()

    for dataset in BEIR_DATASETS:
        dataset = dataset.replace("/", "-")
        fname = f"{input_args.mdir}/{dataset}.json"
        if os.path.exists(fname):
            with open(fname) as file:
                metrics = json.load(file)
            print(dataset, ":")
            print(metrics[dataset])


