import scipy.sparse as sp, numpy as np, torch
from numba import njit, prange

from typing import Optional
from tqdm.auto import tqdm

from xclib.utils.sparse import retain_topk
from xclib.utils import graph

from sugar.core import *

from xcai.metrics import *
from xcai.graph.random_walk import * 

def min_max_normalize(x):
    for i,j in zip(x.indptr, x.indptr[1:]):
        min_val = np.min(x.data[i:j])
        max_val = np.max(x.data[i:j])
        range_val = max_val - min_val

        if range_val == 0:
            x.data[i:j] = np.ones_like(x.data[i:j])
        else:
            x.data[i:j] = (x.data[i:j] - min_val) / range_val

@njit
def sample_from_dist(dist):
    total = dist.sum()
    assert total != 0, "Distribution cannot be zero"

    r = np.random.random() * total
    cum = 0.0
    for i in range(dist.shape[0]):
        cum += dist[i]
        if r <= cum: return i
    return dist.shape[0] - 1

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


@njit(parallel=True, nogil=True)
def _random_walk(
    data_lbl_indices:np.ndarray, 
    data_lbl_indptr:np.ndarray, 
    data_lbl_data:np.ndarray, 
    lbl_data_indices:np.ndarray, 
    lbl_data_indptr:np.ndarray,
    lbl_data_data:np.ndarray,
    reset_dist_indices:np.ndarray, 
    reset_dist_indptr:np.ndarray, 
    reset_dist_data:np.ndarray, 
    seed_node_indices:np.ndarray, 
    seed_node_data:np.ndarray, 
    walk_to:int, 
    p_reset:float, 
    hops_per_step:int, 
    start:int, 
    end:int,
    is_homogeneous:bool,
):
    n_data = end - start
    walk_length = 2*walk_to if is_homogeneous else walk_to

    nbr_idx = np.zeros((n_data, walk_length), dtype=np.int32)
    nbr_data = np.zeros((n_data, walk_length), dtype=np.float32)
    
    for idx in range(0, n_data):
        data_i = seed_node_indices[idx + start]

        dist_idx = seed_node_data[idx + start]
        dist_start, dist_end = reset_dist_indptr[dist_idx], reset_dist_indptr[dist_idx+1]
        dist_data, dist_indices = reset_dist_data[dist_start:dist_end], reset_dist_indices[dist_start:dist_end]

        for walk in np.arange(0, walk_length, 2 if is_homogeneous else 1):
            if np.random.random() < p_reset: data_i = dist_indices[sample_from_dist(dist_data)]
        
            # data --> label

            data_start, data_end = data_lbl_indptr[data_i], data_lbl_indptr[data_i+1]
            if data_start == data_end: continue
                
            lbl_i = sample_weighted(data_lbl_indices, data_lbl_data, data_start, data_end)
            if lbl_i == -1: continue

            if hops_per_step == 1 or is_homogeneous: 
                nbr_idx[idx, walk] = lbl_i
                nbr_data[idx, walk] = 1
            
            if is_homogeneous: walk += 1

            # label --> data
            
            lbl_start, lbl_end = lbl_data_indptr[lbl_i], lbl_data_indptr[lbl_i+1]
            if lbl_start == lbl_end: continue

            data_i = sample_weighted(lbl_data_indices, lbl_data_data, lbl_start, lbl_end)
            if data_i == -1: continue
            
            if hops_per_step == 2 or is_homogeneous: 
                nbr_idx[idx, walk] = data_i
                nbr_data[idx, walk] = 1
            
    return nbr_idx.flatten(), nbr_data.flatten()


class PrunedWalk(graph.RandomWalk):
    
    def __init__(self, data_lbl:sp.csr_matrix, reset_dist:sp.csr_matrix, seed_nodes:sp.csr_matrix):
        self.data_lbl = data_lbl.tocsr()
        self.data_lbl.sort_indices()
        self.data_lbl.eliminate_zeros()

        self.reset_dist = reset_dist
        self.seed_nodes = seed_nodes

    def simulate(self, walk_to:Optional[int]=100, p_reset:Optional[float]=0.2, k:Optional[int]=None, hops_per_step:Optional[int]=2, 
                 b_size:Optional[int]=1000, is_homogeneous:Optional[bool]=False):
        assert hops_per_step == 1 or hops_per_step == 2, f"Invalid hops per step: {hops_per_step}"
        
        data_lbl_indices, data_lbl_indptr, data_lbl_data = self.data_lbl.indices, self.data_lbl.indptr, self.data_lbl.data
        
        lbl_data = self.data_lbl.transpose().tocsr()
        lbl_data.sort_indices()
        lbl_data.eliminate_zeros()

        lbl_data_indices, lbl_data_indptr, lbl_data_data = lbl_data.indices, lbl_data.indptr, lbl_data.data

        n_data = self.data_lbl.shape[0]
        n_lbl = self.data_lbl.shape[hops_per_step % 2]

        reset_dist_indices, reset_dist_indptr, reset_dist_data = self.reset_dist.indices, self.reset_dist.indptr, self.reset_dist.data
        seed_node_indices, seed_node_indptr, seed_node_data = self.seed_nodes.indices, self.seed_nodes.indptr, self.seed_nodes.data

        n_start = len(seed_node_indices)
        
        walks = list()
        for idx in tqdm(range(0, n_start, b_size)):
            start, end = idx, min(idx+b_size, n_start)
            cols, data = _random_walk(data_lbl_indices, data_lbl_indptr, data_lbl_data, lbl_data_indices, lbl_data_indptr, lbl_data_data, reset_dist_indices, 
                                      reset_dist_indptr, reset_dist_data, seed_node_indices, seed_node_data, walk_to, p_reset, hops_per_step, 
                                      start=start, end=end, is_homogeneous=is_homogeneous)
            
            rows = np.arange(end-start).reshape(-1, 1)
            rows = np.repeat(rows, 2 * walk_to if is_homogeneous else walk_to, axis=1).flatten()
            
            walk = sp.coo_matrix((data, (rows, cols)), dtype=np.float32, shape=(end-start, n_lbl))
            walk.sum_duplicates()
            walk = walk.tocsr()
            walk.sort_indices()
            
            if k is not None: walk = xs.retain_topk(walk, k=k).tocsr()
                
            walks.append(walk)
            del rows, cols
        walks = sp.vstack(walks, "csr")

        final_walks = list()
        for i, j in zip(seed_node_indptr, seed_node_indptr[1:]):
            final = walks[i:j].sum(axis=0)
            final /= final.sum()
            final_walks.append(sp.csr_matrix(final))
        return sp.vstack(final_walks, "csr")


def get_seed_nodes(reset_dist:sp.csr_matrix, num_seed_nodes:Optional[int]=None):
    seed_indices, seed_data, seed_indptr = [], [], [0]
    for idx, (i, j) in enumerate(zip(reset_dist.indptr, reset_dist.indptr[1:])):
        seed_indices.extend(reset_dist.indices[i:j])
        seed_data.extend([idx] * (j - i))

        if num_seed_nodes is not None:
            seed_indices.extend(np.random.permutation(reset_dist.shape[1])[:num_seed_nodes])
            seed_data.extend([idx] * num_seed_nodes)

        seed_indptr.append(len(seed_indices))

    return sp.csr_matrix((seed_data, seed_indices, seed_indptr), dtype=np.int64)


def personalized_node_rank(matrix:sp.csr_matrix, reset_dist:sp.csr_matrix, row_head_thresh:Optional[int]=500, col_head_thresh:Optional[int]=500, 
                           p_reset:Optional[float]=0.8, walk_length:Optional[int]=400, topk:Optional[int]=None, batch_size:Optional[int]=1024, 
                           num_seed_nodes:Optional[int]=None, is_homogeneous:Optional[bool]=False):
    matrix = matrix.tocsr()

    idxs = np.where(matrix.getnnz(axis=1) > row_head_thresh)[0]
    pruned_matrix = remove_rows(matrix, idxs) if len(idxs) > 0 else matrix
        
    idxs = np.where(matrix.getnnz(axis=0) > col_head_thresh)[0]
    if len(idxs) > 0: pruned_matrix = remove_cols(pruned_matrix, idxs)

    seed_nodes = get_seed_nodes(reset_dist, num_seed_nodes)
    return PrunedWalk(pruned_matrix, reset_dist, seed_nodes).simulate(walk_length, p_reset, topk, 1, batch_size, is_homogeneous)


if __name__ == "__main__":

    # data paths

    data_dir = "/data/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"

    qry_fact_pred = sp.load_npz(f"{data_dir}/predictions/multihop/musique-hipporag/test_facts.npz")
    qry_fact_pred = retain_topk(qry_fact_pred, k=6)
    min_max_normalize(qry_fact_pred)
    qry_fact_pred.eliminate_zeros()

    qry_lbl_pred = sp.load_npz(f"{data_dir}/predictions/multihop/musique-hipporag/test_labels.npz")
    min_max_normalize(qry_lbl_pred)
    qry_lbl_pred.eliminate_zeros()

    ent_top_k, ent_thresh = 200, 0.8
    lbl_weight = 0.5

    # load data

    data_dir = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"

    ent_ent_sim = retain_topk(sp.load_npz(f"{data_dir}/predictions/multihop/musique/entities_entities.npz"), k=ent_top_k)
    ent_ent_sim.data[ent_ent_sim.data < ent_thresh] = 0.0
    ent_ent_sim.eliminate_zeros()

    ent_ent = sp.load_npz("/data/datasets/multihop/musique/XC/entity_entity_X_Y.npz")
    ent_ent = ent_ent + ent_ent_sim

    fact_ent = sp.load_npz("/data/datasets/multihop/musique/XC/entity_fact_X_Y.npz")

    lbl_file = "/home/sasokan/suchith/HippoRAG/reproduce/dataset/musique/raw_data/label.raw.csv"
    all_lbl_file = "/data/datasets/multihop/musique/XC/raw_data/label.raw.csv"
    lbl_ids, lbl_txt = load_raw_file(lbl_file)
    all_lbl_ids, all_lbl_txt = load_raw_file(all_lbl_file)
    all_lbl_txt2idx = {k:i for i,k in enumerate(all_lbl_txt)}
    valid_lbl_idx = [all_lbl_txt2idx[o] for o in lbl_txt]

    lbl_ent = sp.load_npz("/data/datasets/multihop/musique/XC/entity_lbl_X_Y.npz")
    trn_lbl = sp.load_npz("/data/datasets/multihop/musique/XC/trn_X_Y.npz")

    lbl_ent = lbl_ent[valid_lbl_idx]
    trn_lbl = trn_lbl[:, valid_lbl_idx]

    # accumulate scores

    ## entity score

    qry_ent = qry_fact_pred @ fact_ent

    qry_fact_cnt = qry_fact_pred.copy()
    qry_fact_cnt.data[:] = 1.0
    qry_ent_cnt = qry_fact_cnt @ fact_ent

    assert np.all(qry_ent.indices == qry_ent_cnt.indices)
    assert np.all(qry_ent.indptr == qry_ent_cnt.indptr)

    qry_ent.data[:] = qry_ent.data / qry_ent_cnt.data

    ## label score

    qry_lbl_pred = qry_lbl_pred * lbl_weight

    qry_nodes = sp.hstack([qry_ent, qry_lbl_pred])

    # graph construction

    lbl_lbl = trn_lbl.T @ trn_lbl
    
    mat_1 = sp.hstack([ent_ent, lbl_ent.T])
    mat_2 = sp.hstack([lbl_ent, lbl_lbl])
    matrix = sp.vstack([mat_1, mat_2])

    # matrix = normalize_graph(matrix)

    # random walk

    # output = random_walk(matrix, row_head_thresh=100, col_head_thresh=100, walk_length=400, 
    #                      p_reset=0.4, topk=None, batch_size=1024)

    results = personalized_node_rank(matrix, qry_nodes, row_head_thresh=500, col_head_thresh=500, walk_length=400, 
                                     p_reset=0.8, topk=None, batch_size=1024, num_seed_nodes=200, is_homogeneous=True)

    qry_pred = results[:, qry_ent.shape[1]:]

    metric = PrecReclHits(qry_lbl_pred.shape[1], pk=10, rk=200, hk=10, rep_pk=[1, 3, 5, 10], rep_rk=[5, 10, 100, 200], rep_hk=[1, 3, 5, 10])
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



