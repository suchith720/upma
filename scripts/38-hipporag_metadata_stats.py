import os, scipy.sparse as sp, argparse

if __name__ == "__main__":
    
    data_dir = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-002/predictions/beir/"

    for dset in "arguana fiqa msmarco nfcorpus scidocs scifact trec-covid webis-touche2020".split(" "):
        fname = f"{data_dir}/{dset}/test_facts.npz" 
        tst_meta = sp.load_npz(fname)
        print(dset)
        print("Test fact : ", tst_meta.getnnz(axis=0).mean())

        fname = f"/data/datasets/beir/{dset}/XC/hipporag-fact_lbl_X_Y.npz" 
        tst_meta = sp.load_npz(fname)
        print("Label fact : ", tst_meta.getnnz(axis=0).mean())

        if dset == "msmarco":
            fname = f"{data_dir}/{dset}/train_facts.npz" 
            tst_meta = sp.load_npz(fname)

