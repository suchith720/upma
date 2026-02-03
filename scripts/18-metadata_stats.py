import scipy.sparse as sp

if __name__ == "__main__":

    files = [
        "/data/datasets/beir/msmarco/XC/trn_X_Y.npz",
        "/data/datasets/beir/msmarco/XC/substring/conflation_02/substring_trn_X_Y.npz",
        "/data/datasets/beir/msmarco/XC/intent_substring/conflation_01/intent_trn_X_Y.npz",
        "/data/datasets/beir/msmarco/XC/category-gpt-linker_trn_X_Y_conflated-001_conflated-001.npz",
        "/data/datasets/beir/msmarco/XC/entity_gpt_trn_X_Y_conflated.npz",
    ]

    for file in files:
        print(file)
        m = sp.load_npz(file)

        print(f"# Query: {m.shape[0]}")
        print(f"# Metadata: {m.shape[1]}")
        print(f"# queries/metadata: {m.getnnz(axis=0).mean()}")
        print(f"# metadata/query: {m.getnnz(axis=1).mean()}")
        print()


