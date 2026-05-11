import os, torch, joblib, scipy.sparse as sp, numpy as np, logging

from itertools import chain
from hashlib import md5
from tqdm.auto import tqdm
from typing import Optional, List, Tuple, Dict
from collections import defaultdict

from xcai.metrics import *

from igraph import Graph
import igraph as ig

from sugar.core import load_raw_file
from xclib.utils.sparse import retain_topk

logger = logging.getLogger(__name__)

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val == 0: return np.ones_like(x)
    return (x - min_val) / range_val

class HippoRetrieval:

    def __init__(
        self,
    ):
        self.graph = self.initialize_graph()
        self.ent_node_to_chunk_ids = None

    def initialize_graph(self, is_directed_graph:Optional[bool]=False):
        return ig.Graph(directed=is_directed_graph)

    def add_fact_edges(self, lbl_ids:List[str], lbl_triples:List[Tuple]):
        for lbl_id, triples in tqdm(zip(lbl_ids, lbl_triples), total=len(lbl_ids)):
            entities_in_lbl = set()
            for triple in triples:
                triple = tuple(triple)

                node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                    (node_key, node_2_key), 0.0) + 1
                self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                    (node_2_key, node_key), 0.0) + 1

                entities_in_lbl.add(node_key)
                entities_in_lbl.add(node_2_key)

            for node in entities_in_lbl:
                self.ent_node_to_lbl_ids[node] = self.ent_node_to_lbl_ids.get(node, set()).union(set([lbl_id]))

    def add_passage_edges(self, lbl_ids:List[str], lbl_entities:List[List[str]]):
        for lbl_id, entities in tqdm(zip(lbl_ids, lbl_entities), total=len(lbl_ids)):
            for entity in entities:
                node_key = compute_mdhash_id(entity, prefix="entity-")
                self.node_to_node_stats[(lbl_id, node_key)] = 1.0

    def add_synonymy_edges(self, entity_ids:List, entity_similarity:sp.csr_matrix):
        for i in tqdm(range(len(entity_ids)), total=len(entity_ids)):
            for score, j in zip(entity_similarity[i].data, entity_similarity[i].indices):
                node_key = entity_ids[i]
                node_2_key = entity_ids[j] 
                if node_key != node_2_key: self.node_to_node_stats[(node_key, node_2_key)] = score

    def augment_graph(self, ent_ids:List, lbl_ids:List):
        self.add_new_nodes(ent_ids, lbl_ids)
        self.add_new_edges()

    def add_new_nodes(self, ent_ids:List, lbl_ids:List):
        node_ids = ent_ids + lbl_ids
        self.graph.add_vertices(n=len(node_ids), attributes={"name": node_ids})

    def add_new_edges(self):
        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def index(self, lbl_ids:List[str], lbl_triples:List[str], ent_ids:Optional[List]=None, ent_sim:Optional[sp.csr_matrix]=None, 
              graph_file:Optional[str]=None):

        if graph_file is not None and os.path.exists(graph_file):
            self.graph = joblib.load(graph_file)
        else:
            self.node_to_node_stats = {}
            self.ent_node_to_lbl_ids = {}

            self.add_fact_edges(lbl_ids, lbl_triples)

            lbl_entities = [chain(*[[k[0], k[2]] for k in o]) for o in lbl_triples]
            lbl_entities = [set(o) for o in lbl_entities]
            self.add_passage_edges(lbl_ids, lbl_entities)

            if ent_sim is not None: self.add_synonymy_edges(ent_ids, ent_sim)

            self.augment_graph(ent_ids, lbl_ids)

            if graph_file is not None: joblib.dump(self.graph, graph_file)

    def get_top_k_weights(
        self,
        link_top_k: int,
        all_phrase_weights: np.ndarray,
        linking_score_map: Dict[str, float]
    ):
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert (all_phrase_weights > 0).sum() == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map


    def graph_search_with_fact_entities(
        self, 
        link_top_k: int,
        top_k_facts: List[Tuple], 
        top_k_fact_scores: List[float],
        top_k_fact_indices: List[str],
        qry_lbl: sp.csr_matrix,
        passage_node_weight: float = 0.05
    ):
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        for rank, (f, score) in enumerate(zip(top_k_facts, top_k_fact_scores)):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_idx = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_idx is not None:
                    if len(self.ent_node_to_lbl_ids.get(phrase_key, set())) > 0:
                        score /= len(self.ent_node_to_lbl_ids[phrase_key])

                    phrase_weights[phrase_idx] += score
                    number_of_occurs[phrase_idx] += 1

        mask = number_of_occurs != 0
        phrase_weights[mask] = phrase_weights[mask] / number_of_occurs[mask]

        if link_top_k:
            sort_idxs = np.argsort(phrase_weights)[::-1][:link_top_k]
            all_idxs = sp.csr_matrix(phrase_weights).indices
            for i in set(all_idxs).difference(sort_idxs): phrase_weights[i] = 0.0

        #Get passage scores according to chosen dense retrieval model
        sort_idxs = np.argsort(qry_lbl.data)[::-1]
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = qry_lbl.indices[sort_idxs], qry_lbl.data[sort_idxs] 
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        #Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=0.5)

        assert len(ppr_sorted_doc_ids) == len(self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def run_ppr(
        self,
        reset_prob: np.ndarray,
        damping: float =0.5
    ):
        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(max(list(self.node_name_to_vertex_idx.values())) + 1),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

    def retrieve(
        self,
        qry_fact: sp.csr_matrix,
        qry_lbl:sp.csr_matrix,
        ent_ids:List,
        lbl_ids:List,
        lbl_txt:List,
        facts:List,
        num_to_retrieve:Optional[int]=200,
        link_top_k:Optional[int]=5,
        passage_node_weight:Optional[float]=0.05,
    ):
        retrieval_results = []

        igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
        self.node_name_to_vertex_idx = igraph_name_to_idx

        self.entity_node_keys, self.passage_node_keys, self.passage_node_text = ent_ids, lbl_ids, lbl_txt
        self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys]
        self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys]

        for i in tqdm(range(qry_fact.shape[0]), desc="Retrieving", total=qry_fact.shape[0]):
            scores = min_max_normalize(qry_fact[i].data)
            sort_idx = np.argsort(scores)[::-1][:link_top_k]

            top_k_fact_indices, top_k_fact_scores = qry_fact[i].indices[sort_idx], scores[sort_idx]
            top_k_facts = [facts[i] for i in top_k_fact_indices]
            
            sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(link_top_k=link_top_k,
                                                                                     top_k_facts=top_k_facts,
                                                                                     top_k_fact_scores=top_k_fact_scores,
                                                                                     top_k_fact_indices=top_k_fact_indices,
                                                                                     qry_lbl=qry_lbl[i],
                                                                                     passage_node_weight=passage_node_weight)

            # top_k_docs = [self.passage_node_text[idx] for idx in sorted_doc_ids[:num_to_retrieve]]
            retrieval_results.append((sorted_doc_ids[:num_to_retrieve], sorted_doc_scores[:num_to_retrieve]))

        return retrieval_results

if __name__ == "__main__":
    dataset_type = "musique-hipporag"
    sample = False

    # load label information

    if dataset_type == "musique-hipporag":
        all_lbl_file = "/data/datasets/multihop/musique/XC/raw_data/label.raw.csv"
        lbl_file = "/home/sasokan/suchith/HippoRAG/reproduce/dataset/musique/raw_data/label.raw.csv"

    elif dataset_type == "musique":
        lbl_file = "/data/datasets/multihop/musique/XC/raw_data/label.raw.csv"

    else:
        raise ValueError("Invalid dataset type.")

    lbl_ids, lbl_txt = load_raw_file(lbl_file)
    if dataset_type == "musique-hipporag":
        all_lbl_ids, all_lbl_txt = load_raw_file(all_lbl_file)
        all_lbl_txt2idx = {k:i for i,k in enumerate(all_lbl_txt)}
        valid_lbl_idx = [all_lbl_txt2idx[o] for o in lbl_txt]
    
    # metadata files -- triples, facts, entities

    ent_file = "/data/datasets/multihop/musique/XC/raw_data/entity.raw.csv"
    lbl_triples_file = "/data/datasets/multihop/musique/XC/raw_data/label_triples.joblib"
    fact_file = "/data/datasets/multihop/musique/XC/raw_data/triples.joblib"

    ent_ids, ent_txt = load_raw_file(ent_file)
    facts = joblib.load(fact_file)

    lbl_triples = joblib.load(lbl_triples_file)
    if dataset_type == "musique-hipporag":
        assert len(lbl_triples) == len(all_lbl_txt2idx)
        lbl_triples = [lbl_triples[i] for i in valid_lbl_idx]

    # model predictions -- facts and labels

    data_dir = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"
    if dataset_type == "musique-hipporag":
        qry_fact = sp.load_npz(f"{data_dir}/predictions/multihop/musique-hipporag/test_facts.npz")
        qry_lbl = sp.load_npz(f"{data_dir}/predictions/multihop/musique-hipporag/test_labels.npz")
    else:
        qry_fact = sp.load_npz(f"{data_dir}/predictions/multihop/musique/test_facts.npz")
        qry_lbl = sp.load_npz(f"{data_dir}/predictions/multihop/musique/test_labels.npz")

    # entity-entity similarity

    data_dir = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"
    ent_ent = retain_topk(sp.load_npz(f"{data_dir}/predictions/multihop/musique/entities_entities.npz"), k=100)
    ent_ent.data[ent_ent.data < 0.8] = 0.0
    ent_ent.eliminate_zeros()

    # compute hashes

    lbl_hash_ids = [compute_mdhash_id(content=o, prefix="label-") for o in lbl_txt]
    ent_hash_ids = [compute_mdhash_id(content=o, prefix="entity-") for o in ent_txt]

    # sampling

    n_samples = 1000

    if sample:
        lbl_hash_ids = lbl_hash_ids[:n_samples]
        lbl_triples = lbl_triples[:n_samples]

        entities = [chain(*[[k[0], k[2]] for k in o]) for o in lbl_triples]
        entities = chain(*[list(set(o)) for o in entities])
        ent_ids2idx = {k:i for i,k in enumerate(ent_hash_ids)}
        idxs = [ent_ids2idx[compute_mdhash_id(content=o, prefix="entity-")] for o in entities]

        ent_hash_ids = [ent_hash_ids[i] for i in idxs]
        ent_ent = ent_ent[idxs][: , idxs]

        lbl_txt = lbl_txt[:n_samples]
        qry_lbl = qry_lbl[:, :n_samples]

        facts_ids2idx = {k:i for i,k in enumerate(facts)}
        facts = list(set([tuple(t) for o in lbl_triples for t in o]))
        idxs = [facts_ids2idx[f] for f in facts]

        qry_fact = qry_fact[:, idxs]

        idxs = np.where(np.logical_and(qry_fact.getnnz(axis=1) > 0, qry_lbl.getnnz(axis=1) > 0))[0]
        qry_fact, qry_lbl = qry_fact[idxs], qry_lbl[idxs]

    # HipppoRAG retrieval

    graph_file = "reproduce/graph_musique.joblib" if dataset_type == "musique" else "reproduce/graph_musique-hipporag.joblib"

    module = HippoRetrieval()
    print("Indexing ...")
    module.index(lbl_hash_ids, lbl_triples, ent_hash_ids, ent_ent, graph_file)

    print("Graph walk ...")
    results = module.retrieve(qry_fact, qry_lbl, ent_hash_ids, lbl_hash_ids, lbl_txt, facts, num_to_retrieve=200)

    # save score matrix

    data, indices, indptr = [], [], [0]
    for i, sc in results:
        data.extend(sc)
        indices.extend(i)
        indptr.append(len(indices))
    qry_pred = sp.csr_matrix((data, indices, indptr), dtype=np.float32, shape=qry_lbl.shape)

    save_dir = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"
    if dataset_type == "musique-hipporag":
        sp.save_npz(f"{save_dir}/predictions/multihop/musique-hipporag/test_labels_hipporag.npz", qry_pred)
    else:
        sp.save_npz(f"{save_dir}/predictions/multihop/musique/test_labels_hipporag.npz", qry_pred)

    # compute metrics

    metric = PrecReclHits(len(lbl_txt), pk=10, rk=200, hk=10, rep_pk=[1, 3, 5, 10], rep_rk=[5, 10, 100, 200], 
                          rep_hk=[1, 3, 5, 10])
    o = {
        'pred_idx': torch.tensor(qry_pred.indices, dtype=torch.int64),
        'pred_score': torch.tensor(qry_pred.data, dtype=torch.float32),
        'pred_ptr': torch.tensor([p-q for p,q in zip(qry_pred.indptr[1:], qry_pred.indptr)], dtype=torch.int64),
    }

    if dataset_type == "musique-hipporag":
        gt = sp.load_npz("/home/sasokan/suchith/HippoRAG/reproduce/dataset/musique/tst_X_Y.npz")
    else:
        gt = sp.load_npz("/data/datasets/multihop/musique/XC/tst_X_Y.npz")
    assert gt.shape == qry_pred.shape

    t = {
        'targ_idx': torch.tensor(gt.indices, dtype=torch.int64),
        'targ_score': torch.tensor(gt.data, dtype=torch.float32),
        'targ_ptr': torch.tensor([p-q for p,q in zip(gt.indptr[1:], gt.indptr)], dtype=torch.int64),
    }
    value = metric(**o, **t)
    print(value)

