import scipy.sparse as sp, numpy as np

from sugar.core import *

if __name__ == "__main__":
    trn_lbl = sp.load_npz("/data/datasets/multihop/musique/XC/trn_X_Y.npz")
    trn_ids, trn_txt = load_raw_file("/data/datasets/multihop/musique/XC/raw_data/train.raw.csv")
    lbl_ids, lbl_txt = load_raw_file("/data/datasets/multihop/musique/XC/raw_data/label.raw.csv")

    reform_trn_ids, reform_trn_txt = [], []

    data, indices, indptr = [], [], [0]
    for i in range(trn_lbl.shape[0]):
        query2label = dict()

        query2label[(trn_txt[i], trn_ids[i])] = [(p,q) for p,q in zip(trn_lbl[i].data, trn_lbl[i].indices)]
        stack = [(trn_txt[i], trn_ids[i])]

        while len(stack):
            query, identifier = stack.pop()
            info = set(query2label[(query, identifier)])

            if len(info) > 1:
                for i in info:
                    reform_query = query + " [SEP] " + lbl_txt[i[1]]
                    reform_id = "qry-" + str(identifier) + "_lbl-" + str(lbl_ids[i[1]])

                    query2label[(reform_query, reform_id)] = list(info.difference([i]))
                    stack.append((reform_query, reform_id))

        for k,v in query2label.items():
            d, i = list(zip(*v))
            reform_trn_txt.append(k[0])
            reform_trn_ids.append(k[1])
            data.extend(d)
            indices.extend(i)
            indptr.append(len(data))

    reform_trn_lbl = sp.csr_matrix((data, indices, indptr), shape=(len(reform_trn_txt), len(lbl_txt)), dtype=np.float32)
    sp.save_npz("/data/datasets/multihop/musique/XC/trn-reform_X_Y.npz", reform_trn_lbl)
    save_raw_file("/data/datasets/multihop/musique/XC/raw_data/train-reform.raw.csv", reform_trn_ids, reform_trn_txt)

