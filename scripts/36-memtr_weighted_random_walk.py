import scipy.sparse as sp, numpy as np, torch, os, joblib
from numba import njit, prange

from typing import Optional
from tqdm.auto import tqdm

from xclib.utils.sparse import retain_topk
from xclib.utils import graph

from sugar.core import *

from xcai.metrics import *
from xcai.graph.random_walk import * 

from hashlib import md5

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()

def min_max_normalize(x):
    for i,j in zip(x.indptr, x.indptr[1:]):
        if i == j: continue

        min_val = np.min(x.data[i:j])
        max_val = np.max(x.data[i:j])
        range_val = max_val - min_val

        if range_val == 0:
            x.data[i:j] = np.ones_like(x.data[i:j])
        else:
            x.data[i:j] = (x.data[i:j] - min_val) / range_val

@njit
def sample_weighted(indices, data, start, end):
    total = data[start:end].sum()
    if total == 0: return -1
    r = np.random.random() * total

    cum = 0.0
    for i in range(start, end):
        cum += data[i]
        if r <= cum: return indices[i]
    return indices[end - 1]


# @njit(parallel=True, nogil=True)
# def _random_walk(
#     data_lbl_indices:np.ndarray, 
#     data_lbl_indptr:np.ndarray, 
#     data_lbl_data:np.ndarray, 
#     lbl_data_indices:np.ndarray, 
#     lbl_data_indptr:np.ndarray,
#     lbl_data_data:np.ndarray,
#     walk_to:int, 
#     p_reset:float, 
#     hops_per_step:int, 
#     start:int, 
#     end:int,
#     is_homogeneous:bool,
# ):
#     n_data = end - start
#     walk_length = 2*walk_to if is_homogeneous else walk_to
# 
#     nbr_idx = np.zeros((n_data, walk_length), dtype=np.int32)
#     nbr_data = np.zeros((n_data, walk_length), dtype=np.float32)
#     
#     for idx in range(0, n_data):
#         data_i = idx + start
# 
#         for walk in np.arange(0, walk_length, 2 if is_homogeneous else 1):
#             if np.random.random() < p_reset: data_i = idx + start 
#         
#             # data --> label
# 
#             data_start, data_end = data_lbl_indptr[data_i], data_lbl_indptr[data_i+1]
#             if data_start == data_end: continue
#                 
#             lbl_i = sample_weighted(data_lbl_indices, data_lbl_data, data_start, data_end)
#             if lbl_i == -1: continue
# 
#             if hops_per_step == 1 or is_homogeneous: 
#                 nbr_idx[idx, walk] = lbl_i
#                 nbr_data[idx, walk] = 1
#             
#             if is_homogeneous: walk += 1
# 
#             # label --> data
#             
#             lbl_start, lbl_end = lbl_data_indptr[lbl_i], lbl_data_indptr[lbl_i+1]
#             if lbl_start == lbl_end: continue
# 
#             data_i = sample_weighted(lbl_data_indices, lbl_data_data, lbl_start, lbl_end)
#             if data_i == -1: continue
#             
#             if hops_per_step == 2 or is_homogeneous: 
#                 nbr_idx[idx, walk] = data_i
#                 nbr_data[idx, walk] = 1
#             
#     return nbr_idx.flatten(), nbr_data.flatten()


@njit(parallel=True, nogil=True)
def _random_walk(
    data_lbl_indices:np.ndarray, 
    data_lbl_indptr:np.ndarray, 
    data_lbl_data:np.ndarray, 
    lbl_data_indices:np.ndarray, 
    lbl_data_indptr:np.ndarray,
    lbl_data_data:np.ndarray,
    walk_to:int, 
    p_reset:float, 
    start:int, 
    end:int,
):
    n_data = end - start

    nbr_idx = np.zeros((n_data, walk_to), dtype=np.int32)
    nbr_data = np.zeros((n_data, walk_to), dtype=np.float32)
    
    for idx in range(0, n_data):
        data_i = idx + start

        for walk in np.arange(0, walk_to):
            if np.random.random() < p_reset: data_i = idx + start 
        
            # data --> label

            data_start, data_end = data_lbl_indptr[data_i], data_lbl_indptr[data_i+1]
            if data_start == data_end: continue
                
            data_i = sample_weighted(data_lbl_indices, data_lbl_data, data_start, data_end)
            if data_i == -1: continue

            nbr_idx[idx, walk] = data_i
            nbr_data[idx, walk] = 1
            
    return nbr_idx.flatten(), nbr_data.flatten()


class PrunedWalk(graph.RandomWalk):
    
    def __init__(self, data_lbl:sp.csr_matrix):
        self.data_lbl = data_lbl.tocsr()
        self.data_lbl.sort_indices()
        self.data_lbl.eliminate_zeros()

    # def simulate(self, walk_to:Optional[int]=100, p_reset:Optional[float]=0.2, k:Optional[int]=None, hops_per_step:Optional[int]=2, 
    #              b_size:Optional[int]=1000, is_homogeneous:Optional[bool]=False):
    #     assert hops_per_step == 1 or hops_per_step == 2, f"Invalid hops per step: {hops_per_step}"
    #     
    #     data_lbl_indices, data_lbl_indptr, data_lbl_data = self.data_lbl.indices, self.data_lbl.indptr, self.data_lbl.data
    #     
    #     lbl_data = self.data_lbl.transpose().tocsr()
    #     lbl_data.sort_indices()
    #     lbl_data.eliminate_zeros()

    #     lbl_data_indices, lbl_data_indptr, lbl_data_data = lbl_data.indices, lbl_data.indptr, lbl_data.data

    #     n_data = self.data_lbl.shape[0]
    #     n_lbl = self.data_lbl.shape[hops_per_step % 2]

    #     walks = list()
    #     for idx in tqdm(range(0, n_data, b_size)):
    #         start, end = idx, min(idx+b_size, n_data)
    #         cols, data = _random_walk(data_lbl_indices, data_lbl_indptr, data_lbl_data, lbl_data_indices, lbl_data_indptr, lbl_data_data,
    #                                   walk_to, p_reset, hops_per_step, start=start, end=end, is_homogeneous=is_homogeneous)
    #         
    #         rows = np.arange(end-start).reshape(-1, 1)
    #         rows = np.repeat(rows, 2 * walk_to if is_homogeneous else walk_to, axis=1).flatten()
    #         
    #         walk = sp.coo_matrix((data, (rows, cols)), dtype=np.float32, shape=(end-start, n_lbl))
    #         walk.sum_duplicates()
    #         walk = walk.tocsr()
    #         walk.sort_indices()
    #         
    #         if k is not None: walk = xs.retain_topk(walk, k=k).tocsr()
    #             
    #         walks.append(walk)
    #         del rows, cols
    #         
    #     return sp.vstack(walks, "csr")

    def simulate(self, walk_to:Optional[int]=100, p_reset:Optional[float]=0.2, k:Optional[int]=None, b_size:Optional[int]=1000):

        data_lbl_indices, data_lbl_indptr, data_lbl_data = self.data_lbl.indices, self.data_lbl.indptr, self.data_lbl.data
        
        lbl_data = self.data_lbl.transpose().tocsr()
        lbl_data.sort_indices()
        lbl_data.eliminate_zeros()

        lbl_data_indices, lbl_data_indptr, lbl_data_data = lbl_data.indices, lbl_data.indptr, lbl_data.data

        n_data = self.data_lbl.shape[0]
        n_lbl = self.data_lbl.shape[1]

        walks = list()
        for idx in tqdm(range(0, n_data, b_size)):
            start, end = idx, min(idx+b_size, n_data)
            cols, data = _random_walk(data_lbl_indices, data_lbl_indptr, data_lbl_data, lbl_data_indices, lbl_data_indptr, lbl_data_data,
                                      walk_to, p_reset, start=start, end=end)
            
            rows = np.arange(end-start).reshape(-1, 1)
            rows = np.repeat(rows, walk_to, axis=1).flatten()
            
            walk = sp.coo_matrix((data, (rows, cols)), dtype=np.float32, shape=(end-start, n_lbl))
            walk.sum_duplicates()
            walk = walk.tocsr()
            walk.sort_indices()
            
            if k is not None: walk = xs.retain_topk(walk, k=k).tocsr()
                
            walks.append(walk)
            del rows, cols
            
        return sp.vstack(walks, "csr")

        
def weighted_random_walk(matrix:sp.csr_matrix, row_head_thresh:Optional[int]=500, col_head_thresh:Optional[int]=500, walk_length:Optional[int]=400,
                         p_reset:Optional[float]=0.8, topk:Optional[int]=10, batch_size:Optional[int]=1023, is_homogeneous:Optional[bool]=False):
    matrix = matrix.tocsr()
    
    # idxs = np.where(matrix.getnnz(axis=1) > row_head_thresh)[0]
    # pruned_matrix = remove_rows(matrix, idxs) if len(idxs) > 0 else matrix
    #   
    # idxs = np.where(matrix.getnnz(axis=0) > col_head_thresh)[0]
    # if len(idxs) > 0: pruned_matrix = remove_cols(pruned_matrix, idxs)
    
    # return PrunedWalk(matrix).simulate(walk_length, p_reset, topk, 2, batch_size, is_homogeneous=is_homogeneous)

    return PrunedWalk(matrix).simulate(walk_length, p_reset, topk, batch_size)


def make_bidirectional_graph(matrix:sp.csr_matrix):
    assert matrix.shape[0] == matrix.shape[1]
    rows, cols = matrix.nonzero()
    matrix_t = matrix.transpose().tocsr(copy=True)
    matrix_t[rows, cols] = 0.0
    matrix_t.eliminate_zeros()
    return matrix + matrix_t


if __name__ == "__main__":

    # test predictions -- facts and labels

    data_dir = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"

    qry_fact_pred = sp.load_npz(f"{data_dir}/predictions/multihop/musique-hipporag/test_facts.npz")
    min_max_normalize(qry_fact_pred)
    qry_fact_pred = retain_topk(qry_fact_pred, k=5)
    qry_fact_pred.eliminate_zeros()

    qry_lbl_pred = sp.load_npz(f"{data_dir}/predictions/multihop/musique-hipporag/test_labels.npz")
    min_max_normalize(qry_lbl_pred)
    qry_lbl_pred.eliminate_zeros()

    # entity data

    ent_top_k, ent_thresh = 100, 0.8

    ## entity similarity

    ent_ent_sim = retain_topk(sp.load_npz(f"{data_dir}/predictions/multihop/musique/entities_entities.npz"), k=ent_top_k)
    ent_ent_sim.data[ent_ent_sim.data < ent_thresh] = 0.0
    ent_ent_sim.setdiag(0)
    ent_ent_sim.eliminate_zeros()

    ## entity graph

    ent_ent = sp.load_npz("/data/datasets/multihop/musique/XC/entity_entity_X_Y.npz")

    ent_ent = ent_ent + ent_ent_sim
    ent_ent = ent_ent + ent_ent.transpose() 

    # load data -- fact to entity, label to entity, train to label

    ## get valid label indices

    lbl_file = "/home/sasokan/suchith/HippoRAG/reproduce/dataset/musique/raw_data/label.raw.csv"
    all_lbl_file = "/data/datasets/multihop/musique/XC/raw_data/label.raw.csv"
    lbl_ids, lbl_txt = load_raw_file(lbl_file)
    all_lbl_ids, all_lbl_txt = load_raw_file(all_lbl_file)
    all_lbl_txt2idx = {k:i for i,k in enumerate(all_lbl_txt)}
    valid_lbl_idx = [all_lbl_txt2idx[o] for o in lbl_txt]

    ## fact to entity

    fact_ent = sp.load_npz("/data/datasets/multihop/musique/XC/entity_fact_X_Y.npz")
    fact_ent.data[:] = 1.0

    ## label to entity

    lbl_ent = sp.load_npz("/data/datasets/multihop/musique/XC/entity_lbl_X_Y.npz")
    lbl_ent.data[:] = 1.0
    lbl_ent = lbl_ent[valid_lbl_idx]

    ## train to label

    trn_lbl = sp.load_npz("/data/datasets/multihop/musique/XC/trn_X_Y.npz")
    trn_lbl = trn_lbl[:, valid_lbl_idx]

    # accumulate scores

    ## query-entity score

    qry_ent = qry_fact_pred @ fact_ent

    qry_fact_cnt = qry_fact_pred.copy()
    qry_fact_cnt.data[:] = 1.0
    qry_ent_cnt = qry_fact_cnt @ fact_ent

    assert np.all(qry_ent.indices == qry_ent_cnt.indices)
    assert np.all(qry_ent.indptr == qry_ent_cnt.indptr)

    qry_ent.data[:] = qry_ent.data / qry_ent_cnt.data
    
    ent_per_lbl = lbl_ent.getnnz(axis=0)
    ent_per_lbl[ent_per_lbl == 0] = 1.0

    qry_ent = qry_ent.multiply(1.0 / ent_per_lbl).tocsr()
    qry_ent = retain_topk(qry_ent, k=5)

    ## query-label score

    lbl_weight = 0.05
    qry_nodes = sp.hstack([qry_ent, qry_lbl_pred * lbl_weight])

    # graph construction

    lbl_lbl = sp.csr_matrix((trn_lbl.shape[1], trn_lbl.shape[1]))

    # load graph igraph

    g = joblib.load("reproduce/graph_musique-hipporag.joblib")
    graph = g.get_adjacency_sparse(attribute="weight")
    vertex_txt = g.vs["name"]
    lbl_hash_ids = [compute_mdhash_id(content=o, prefix="label-") for o in lbl_txt]
    assert np.all([p == q for p,q in zip(vertex_txt[-len(lbl_hash_ids):], lbl_hash_ids)])

    qry_nodes_2 = sp.load_npz("/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/multihop/musique-hipporag/node_weights.npz")
    
    mat_1 = sp.hstack([ent_ent, lbl_ent.T])
    mat_2 = sp.hstack([lbl_ent, lbl_lbl])
    matrix = sp.vstack([mat_1, mat_2])

    walk_file = None # "/data/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/multihop/musique-hipporag/walks.npz"
    if walk_file is not None and os.path.exists(walk_file):
        walks = sp.load_npz(walk_file)
    else:
        walks = weighted_random_walk(graph, row_head_thresh=500, col_head_thresh=500, walk_length=400, 
                                     p_reset=0.5, topk=None, batch_size=1024, is_homogeneous=True)
        if walk_file is not None: sp.save_npz(walk_file, walks)

    # # compute scores

    # walk_length = walks.sum(axis=1)
    # walk_length[walk_length == 0] = 1
    # walks = walks / walk_length

    # qry_pred = qry_nodes @ walks
    qry_pred = qry_nodes_2 @ walks

    qry_pred = qry_pred[:, qry_ent.shape[1]:]

    # compute metrics

    metric = PrecReclHits(qry_lbl_pred.shape[1], pk=10, rk=200, hk=10, rep_pk=[1, 3, 5, 10], rep_rk=[5, 10, 100, 200], 
                          rep_hk=[1, 3, 5, 10])
    o = {
        'pred_idx': torch.tensor(qry_pred.indices, dtype=torch.int64),
        'pred_score': torch.tensor(qry_pred.data, dtype=torch.float32),
        'pred_ptr': torch.tensor([p-q for p,q in zip(qry_pred.indptr[1:], qry_pred.indptr)], dtype=torch.int64),
    }

    gt = sp.load_npz("/home/sasokan/suchith/HippoRAG/reproduce/dataset/musique/tst_X_Y.npz")
    assert gt.shape == qry_pred.shape
    t = {
        'targ_idx': torch.tensor(gt.indices, dtype=torch.int64),
        'targ_score': torch.tensor(gt.data, dtype=torch.float32),
        'targ_ptr': torch.tensor([p-q for p,q in zip(gt.indptr[1:], gt.indptr)], dtype=torch.int64),
    }
    value = metric(**o, **t)
    print(value)

