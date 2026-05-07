import pandas as pd, ast, scipy.sparse as sp, numpy as np, re, os, logging, joblib, json
from itertools import chain
from typing import List, Optional

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

def text_processing(text:str):
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

def process_metadata(data_meta:sp.csr_matrix, prefix:Optional[str]=None):
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
    
    prefix = "" if prefix is None else f"{prefix}-"
    meta_ids = [f"{prefix}{i}" for i in range(len(vocab))]
    meta_txt = sorted(vocab, key=lambda x: vocab[x])
    return matrix, meta_ids, meta_txt

def create_entity_graph(entity_txt:List, lbl_triples:List):
    entity_ids2idx = {k:i for i,k in enumerate(entity_txt)}
    entity2entity = {}
    for triples in lbl_triples:
        for triple in triples: 
            entity2entity.setdefault(triple[0], []).append(triple[2])

    data, indices, indptr = [], [], [0]
    for entity in entity_txt:
        indices.extend([entity_ids2idx[o] for o in entity2entity.get(entity, [])])
        indptr.append(len(indices))
    data = [1] * len(indices)

    ent_ent = sp.csr_matrix((data, indices, indptr), shape=(len(entity_txt), len(entity_txt)), dtype=np.float32)
    ent_ent.sum_duplicates()
    ent_ent.eliminate_zeros()

    return ent_ent

def create_fact_entity_graph(fact_txt:List, entity_txt:List, lbl_triples:List):
    entity_ids2idx = {k:i for i,k in enumerate(entity_txt)}
    fact2entity = {}
    for triples in lbl_triples:
        for triple in triples:
            fact2entity.setdefault(triple, []).append(triple[0])
            fact2entity.setdefault(triple, []).append(triple[2])

    data, indices, indptr = [], [], [0]
    for fact in fact_txt:
        indices.extend([entity_ids2idx[o] for o in fact2entity.get(fact, [])])
        indptr.append(len(indices))
    data = [1] * len(indices)

    fact_ent = sp.csr_matrix((data, indices, indptr), shape=(len(fact_txt), len(entity_txt)), dtype=np.float32)
    fact_ent.sum_duplicates()
    fact_ent.eliminate_zeros()

    return fact_ent

def get_fact_examples(fact_txt:List, entity_txt:List, lbl_fact_mat:sp.csr_matrix, lbl_entity_mat:sp.csr_matrix):
    lbl_ids, lbl_txt = load_raw_file("/data/datasets/multihop/musique/XC/raw_data/label.raw.csv")
    idxs = np.random.permutation(len(lbl_txt))[:5]
    examples = []
    for i in idxs:
        example = {
            "Label": lbl_txt[i],
            "Facts": " || ".join([fact_txt[o] for o in lbl_fact_mat[i].indices]),
            "Entities": " || ".join([entity_txt[o] for o in lbl_entity_mat[i].indices]),
        }
        examples.append(example)
    with open("/data/datasets/multihop/musique/examples/01-label_fact.json", "w") as file:
        json.dump(examples, file, indent=4)

def get_entity_examples(entity_txt:List, ent_ent:sp.csr_matrix):
    valid_idxs = np.where(ent_ent.getnnz(axis=1) > 0)[0]
    idxs = np.random.permutation(len(valid_idxs))[:5]
    idxs = valid_idxs[idxs]
    examples = []
    for i in idxs:
        example = {
            "Entity": entity_txt[i],
            "Related entities": " || ".join([entity_txt[o] for o in ent_ent[i].indices]),
        }
        examples.append(example)
    with open("/data/datasets/multihop/musique/examples/02-entity_entity.json", "w") as file:
        json.dump(examples, file, indent=4)

if __name__ == "__main__":

    # Load triples

    fname = "/data/datasets/multihop/musique/XC/raw_data/label_triples.joblib"
    if os.path.exists(fname):
        lbl_triples = joblib.load(fname)
    else:
        lbl_ids, lbl_triples = load_generations()
        lbl_triples = preprocess_triples(lbl_triples)
        joblib.dump(lbl_triples, fname)

    # Format facts

    lbl_fact_mat, fact_ids, triples = process_metadata(lbl_triples, prefix="fact")
    fact_txt = [" ".join(o) for o in triples]
    joblib.dump(triples, "/data/datasets/multihop/musique/XC/raw_data/triples.joblib")

    sp.save_npz("/data/datasets/multihop/musique/XC/fact_lbl_X_Y.npz", lbl_fact_mat)
    save_raw_file("/data/datasets/multihop/musique/XC/raw_data/fact.raw.csv", fact_ids, fact_txt)

    # Format entities

    lbl_entities = [chain(*[[k[0], k[2]] for k in o]) for o in lbl_triples]
    lbl_entity_mat, entity_ids, entity_txt = process_metadata(lbl_entities, prefix="entity")

    sp.save_npz("/data/datasets/multihop/musique/XC/entity_lbl_X_Y.npz", lbl_entity_mat)
    save_raw_file("/data/datasets/multihop/musique/XC/raw_data/entity.raw.csv", entity_ids, entity_txt)
    
    ent_ent = create_entity_graph(entity_txt, lbl_triples)
    sp.save_npz("/data/datasets/multihop/musique/XC/entity_entity_X_Y.npz", ent_ent)

    fact_ent = create_fact_entity_graph(triples, entity_txt, lbl_triples)
    sp.save_npz("/data/datasets/multihop/musique/XC/entity_fact_X_Y.npz", fact_ent)

    # examples

    get_fact_examples(fact_txt, entity_txt, lbl_fact_mat, lbl_entity_mat)

    get_entity_examples(entity_txt, ent_ent)


