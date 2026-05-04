import pandas as pd, ast, scipy.sparse as sp, numpy as np, re, os, logging, joblib
from itertools import chain
from typing import List

from sugar.core import load_raw_file, save_raw_file

logger = logging.getLogger(__name__)

def load_generations():
    data_dir = "/data/datasets/multihop/musique/metadata/outputs-22Apr2026"
    output_file = f"{data_dir}/UHRS_Task_label_and_entity.raw.tsv"
    if os.path.exists(output_file):
        df = pd.read_table(output_file)

    else:
        df1 = pd.read_table(f"{data_dir}/outputs-22Apr2026/UHRS_Task_label_and_entity1.raw.tsv")
        df2 = pd.read_table(f"{data_dir}/outputs-22Apr2026/UHRS_Task_label_and_entity2.raw.tsv")
        df = pd.concat([df1, df2], axis=0)

        lblids2triplesidx = {k:i for i,k in enumerate(df["id"].tolist())}
        lbl_ids, lbl_txt = load_raw_file("/data/datasets/multihop/musique/XC/raw_data/label.raw.csv")
        sort_idx = [lblids2triplesidx[k] for k in lbl_ids]
        df = df.iloc[sort_idx]
        df.to_csv(output_file, index=None, sep="\t")

    return df["id"], df["raw_model_response"]

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def flatten_facts(chunk_triples: List[List]) -> List[List]:
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def preprocess_triples(triples: List[List[List]]):
    triples = [o.replace('```json', '').replace('```', '').strip() for o in triples]
    triples = [ast.literal_eval(o) for o in triples]
    triples = [[[i if isinstance(i, str) else " ".join(i) for i in k] for k in o["triples"]] for o in triples]
    triples = [[[text_processing(i) for i in k] for k in o] for o in triples]
    triples = [[[i for i in k if len(i)] for k in o] for o in triples]
    triples = [[tuple(k) for k in o if len(k) == 3] for o in triples]
    return triples

def process_metadata(data_meta):
    vocab = dict()
    data, indices, indptr = [], [], [0]
    for o in data_meta:
        for t in o:
            i = vocab.setdefault(t, len(vocab))
            indices.append(i)
            data.append(1.0)
        indptr.append(len(data))
    matrix = sp.csr_matrix((data, indices, indptr), shape=(len(data_meta), len(vocab)), dtype=np.float32)
    matrix.eliminate_zeros()
    matrix.sum_duplicates()
    meta_ids = list(range(len(vocab)))
    meta_txt = sorted(vocab, key=lambda x: vocab[x])
    return matrix, meta_ids, meta_txt

if __name__ == "__main__":
    fname = "/data/datasets/multihop/musique/XC/raw_data/label_triples.joblib"
    if os.path.exists(fname):
        lbl_triples = joblib.load(fname)
    else:
        lbl_ids, lbl_triples = load_generations()
        lbl_triples = preprocess_triples(lbl_triples)
        joblib.dump(lbl_triples, fname)

    lbl_phrases = [[" ".join(k) for k in o] for o in lbl_triples]
    lbl_phrase_mat, phrase_ids, phrase_txt = process_metadata(lbl_phrases)
    sp.save_npz("/data/datasets/multihop/musique/XC/phrase_lbl_X_Y.npz", lbl_phrase_mat)
    save_raw_file("/data/datasets/multihop/musique/XC/raw_data/phrase.raw.csv", phrase_ids, phrase_txt)

    lbl_entities = [chain(*[[k[0], k[2]] for k in o]) for o in lbl_triples]
    lbl_entity_mat, entity_ids, entity_txt = process_metadata(lbl_entities)
    sp.save_npz("/data/datasets/multihop/musique/XC/entity_lbl_X_Y.npz", lbl_entity_mat)
    save_raw_file("/data/datasets/multihop/musique/XC/raw_data/entity.raw.csv", entity_ids, entity_txt)

    # examples

    lbl_ids, lbl_txt = load_raw_file("/data/datasets/multihop/musique/XC/raw_data/label.raw.csv")
    idxs = np.random.permutation(len(lbl_txt))[:10]
    for i in idxs:
        print("Label: ", lbl_txt[i])
        print("Phrases: ", " || ".join([phrase_txt[o] for o in lbl_phrase_mat[i].indices]))
        print("Entities: ", " || ".join([entity_txt[o] for o in lbl_entity_mat[i].indices]))
        print()
