"""
Helper utilities for processing and formatting HippoRAG metadata.
Provides text processing, triple extraction, sparse matrix construction,
entity graph construction, and example generation.
"""

import ast, json, logging, os, re
import joblib, json_repair, numpy as np, pandas as pd, scipy.sparse as sp

from tqdm.auto import tqdm
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

from sugar.core import load_raw_file, save_raw_file
from xcai.misc import BEIR_DATASETS

logger = logging.getLogger(__name__)

__all__ = [
    "text_processing",
    "flatten_facts",
    "preprocess_triples",
    "process_metadata",
    "create_entity_graph",
    "create_fact_entity_graph",
    "get_fact_examples",
    "get_entity_examples",
]


def text_processing(text: Union[str, List[Any]]) -> Union[str, List[Any]]:
    """Recursively processes string or list of strings by lowercasing,
    removing special characters, and stripping extra whitespace.

    Args:
        text (Union[str, List[Any]]): Text string or list of text strings/objects.

    Returns:
        Union[str, List[Any]]: Cleaned string or nested list of cleaned strings.
    """
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'[^A-Za-z0-9 ]', ' ', text.lower()).strip()


def flatten_facts(chunk_triples: List[List[Tuple[str, str, str]]]) -> List[Tuple[str, str, str]]:
    """Flattens a list of chunked relation triples into a list of unique relation triples.

    Args:
        chunk_triples (List[List[Tuple[str, str, str]]]): Nested lists of triples.

    Returns:
        List[Tuple[str, str, str]]: Unique flat list of triples.
    """
    graph_triples = []
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    return list(set(graph_triples))


def preprocess_triples(triples: List[str]) -> List[List[Tuple[str, str, str]]]:
    """Robustly parses raw text representing JSON blocks of relation triples
    using ast.literal_eval and json_repair for malformed strings.

    Args:
        triples (List[str]): Raw string predictions containing JSON relation triples.

    Returns:
        List[List[Tuple[str, str, str]]]: Parsed, cleaned, and validated list of triples.
    """
    proc_triples = list()
    for idx, o in tqdm(enumerate(triples), total=len(triples)):
        o = o.replace('```json', '').replace('```', '').strip()

        try:
            o = ast.literal_eval(o)
        except Exception:
            try:
                o = json_repair.loads(o)
            except Exception:
                proc_triples.append([])
                continue

        o = o["triples"] if isinstance(o, dict) and "triples" in o else o
        if not isinstance(o, list) or len(o) == 0 or not isinstance(o[0], list):
            proc_triples.append([])
            continue

        # Clean and format triples
        o = [[str(i) for i in k] for k in o]
        o = [[text_processing(i) for i in k] for k in o]
        o = [[i for i in k if len(i)] for k in o]
        o = [tuple(k) for k in o if len(k) == 3]
        proc_triples.append(o)
    return proc_triples


def process_metadata(data_meta: List[List[str]], prefix: Optional[str] = None) -> Tuple[sp.csr_matrix, List[str], List[str]]:
    """Constructs a sparse binary term-document relation matrix from metadata labels,
    returning the matrix, formatted metadata IDs, and unique vocabulary text list.

    Args:
        data_meta (List[List[str]]): List of metadata annotations per document.
        prefix (Optional[str]): Suffix/prefix prefix for output metadata IDs.

    Returns:
        Tuple[sp.csr_matrix, List[str], List[str]]:
            - csr_matrix: Binary sparse matrix indicating metadata presence.
            - List[str]: Unique IDs for each metadata entry.
            - List[str]: Text descriptions for each metadata entry sorted by ID.
    """
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

    prefix_str = "" if prefix is None else f"{prefix}-"
    meta_ids = [f"{prefix_str}{i}" for i in range(len(vocab))]
    meta_txt = sorted(vocab, key=lambda x: vocab[x])
    return matrix, meta_ids, meta_txt


def create_entity_graph(entity_txt: List[str], lbl_triples: List[List[Tuple[str, str, str]]]) -> sp.csr_matrix:
    """Builds a binary sparse matrix representing entity-entity relations
    derived from the generated triples.

    Args:
        entity_txt (List[str]): Unique entity strings corresponding to matrix indices.
        lbl_triples (List[List[Tuple[str, str, str]]]): Triples containing relation links.

    Returns:
        sp.csr_matrix: Symmetric or directed adjacency matrix of size (N_entities, N_entities).
    """
    entity_ids2idx = {k: i for i, k in enumerate(entity_txt)}
    entity2entity = {}
    for triples in lbl_triples:
        for triple in triples:
            entity2entity.setdefault(triple[0], []).append(triple[2])

    data, indices, indptr = [], [], [0]
    for entity in entity_txt:
        indices.extend([entity_ids2idx[o] for o in entity2entity.get(entity, []) if o in entity_ids2idx])
        indptr.append(len(indices))
    data = [1.0] * len(indices)

    ent_ent = sp.csr_matrix((data, indices, indptr), shape=(len(entity_txt), len(entity_txt)), dtype=np.float32)
    ent_ent.sum_duplicates()
    ent_ent.eliminate_zeros()
    return ent_ent


def create_fact_entity_graph(fact_txt: List[Union[str, Tuple[str, str, str]]], entity_txt: List[str],
                             lbl_triples: List[List[Tuple[str, str, str]]]) -> sp.csr_matrix:
    """Builds a binary sparse matrix representing fact-to-entity relations.
    Handles both list of tuples and list of strings representations.

    Args:
        fact_txt (List[Union[str, Tuple[str, str, str]]]): Flat fact representations.
        entity_txt (List[str]): Flat entity string representations.
        lbl_triples (List[List[Tuple[str, str, str]]]): Raw relation triples.

    Returns:
        sp.csr_matrix: Sparse matrix of shape (N_facts, N_entities).
    """
    entity_ids2idx = {k: i for i, k in enumerate(entity_txt)}
    fact2entity = {}
    for triples in lbl_triples:
        for triple in triples:
            fact_tuple = tuple(triple)
            fact_str = " ".join(triple)
            fact2entity.setdefault(fact_tuple, []).extend([triple[0], triple[2]])
            fact2entity.setdefault(fact_str, []).extend([triple[0], triple[2]])

    data, indices, indptr = [], [], [0]
    for fact in fact_txt:
        key = tuple(fact) if isinstance(fact, (list, tuple)) else fact
        indices.extend([entity_ids2idx[o] for o in fact2entity.get(key, []) if o in entity_ids2idx])
        indptr.append(len(indices))
    data = [1.0] * len(indices)

    fact_ent = sp.csr_matrix((data, indices, indptr), shape=(len(fact_txt), len(entity_txt)), dtype=np.float32)
    fact_ent.sum_duplicates()
    fact_ent.eliminate_zeros()
    return fact_ent


def get_fact_examples(example_dir: str, data_dir: str, fact_txt: List[str], entity_txt: List[str],
                      lbl_fact_mat: sp.csr_matrix, lbl_entity_mat: sp.csr_matrix, num_examples: Optional[int] = 20) -> None:
    """Generates random examples showing labels and their corresponding extracted facts
    and entities, saving the result in a JSON file.

    Args:
        example_dir (str): Directory where the output JSON will be saved.
        data_dir (str): Directory containing dataset files.
        fact_txt (List[str]): Fact string descriptions.
        entity_txt (List[str]): Entity string descriptions.
        lbl_fact_mat (sp.csr_matrix): Sparse label-to-fact mapping matrix.
        lbl_entity_mat (sp.csr_matrix): Sparse label-to-entity mapping matrix.
        num_examples (Optional[int]): Number of random examples to generate. Defaults to 20.
    """
    lbl_ids, lbl_txt = load_raw_file(f"{data_dir}/raw_data/label.raw.csv")

    valid_idxs = np.where(lbl_fact_mat.getnnz(axis=1) > 0)[0]
    if len(valid_idxs) == 0:
        return
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

    os.makedirs(example_dir, exist_ok=True)
    with open(f"{example_dir}/01-hipporag_label_fact.json", "w") as file:
        json.dump(examples, file, indent=4)


def get_entity_examples(example_dir: str, entity_txt: List[str], ent_ent: sp.csr_matrix, num_examples: Optional[int] = 20) -> None:
    """Generates random examples showing entities and their related entities,
    saving the result in a JSON file.

    Args:
        example_dir (str): Directory where the output JSON will be saved.
        entity_txt (List[str]): Entity string descriptions.
        ent_ent (sp.csr_matrix): Sparse entity-to-entity mapping matrix.
        num_examples (Optional[int]): Number of random examples to generate. Defaults to 20.
    """
    valid_idxs = np.where(ent_ent.getnnz(axis=1) > 0)[0]
    if len(valid_idxs) == 0:
        return
    idxs = np.random.permutation(len(valid_idxs))[:num_examples]
    idxs = valid_idxs[idxs]

    examples = []
    for i in idxs:
        example = {
            "Entity": entity_txt[i],
            "Related entities": " || ".join([entity_txt[o] for o in ent_ent[i].indices]),
        }
        examples.append(example)

    os.makedirs(example_dir, exist_ok=True)
    with open(f"{example_dir}/02-hipporag_entity_entity.json", "w") as file:
        json.dump(examples, file, indent=4)
