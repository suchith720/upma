import pandas as pd, ast, scipy.sparse as sp, numpy as np, re, os, logging, joblib, json, json_repair

from tqdm.auto import tqdm
from itertools import chain
from typing import List, Optional

from xcai.misc import BEIR_DATASETS
from sugar.core import load_raw_file, save_raw_file

logger = logging.getLogger(__name__)

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
    proc_triples = list()
    for idx, o in tqdm(enumerate(triples), total=len(triples)):
        o = o.replace('```json', '').replace('```', '').strip()

        try:
            o = ast.literal_eval(o)
        except:
            o = json_repair.loads(o)

        o = o["triples"] if isinstance(o, dict) and "triples" in o else o
        if not isinstance(o, list) or len(o) == 0 or not isinstance(o[0], list): 
            proc_triples.append([])
            continue

        o = [[i.__str__() for i in k] for k in o]
        o = [[text_processing(i) for i in k] for k in o]
        o = [[i for i in k if len(i)] for k in o] 
        o = [tuple(k) for k in o if len(k) == 3]
        proc_triples.append(o)
    return proc_triples

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

def get_fact_examples(example_dir:str, data_dir:str, fact_txt:List, entity_txt:List, lbl_fact_mat:sp.csr_matrix, lbl_entity_mat:sp.csr_matrix, num_examples:Optional[int]=20):
    lbl_ids, lbl_txt = load_raw_file(f"{data_dir}/raw_data/label.raw.csv")

    valid_idxs = np.where(lbl_fact_mat.getnnz(axis=1) > 0)[0]
    idxs = np.random.permutation(len(valid_idxs))[:num_examples]
    idxs = valid_idxs[idxs]

    examples = []
    for i in idxs:
        example = {
            "Label": lbl_txt[i],
            "Facts": " || ".join([fact_txt[o] for o in lbl_fact_mat[i].indices]),
            "Entities": " || ".join([entity_txt[o] for o in lbl_entity_mat[i].indices]),
        }
        examples.append(example)
    with open(f"{example_dir}/01-hipporag_label_fact.json", "w") as file:
        json.dump(examples, file, indent=4)

def get_entity_examples(example_dir:str, entity_txt:List, ent_ent:sp.csr_matrix, num_examples:Optional[int]=20):
    valid_idxs = np.where(ent_ent.getnnz(axis=1) > 0)[0]
    idxs = np.random.permutation(len(valid_idxs))[:num_examples]
    idxs = valid_idxs[idxs]
    examples = []
    for i in idxs:
        example = {
            "Entity": entity_txt[i],
            "Related entities": " || ".join([entity_txt[o] for o in ent_ent[i].indices]),
        }
        examples.append(example)
    with open(f"{example_dir}/02-hipporag_entity_entity.json", "w") as file:
        json.dump(examples, file, indent=4)

