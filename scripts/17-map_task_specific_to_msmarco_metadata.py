import scipy.sparse as sp, numpy as np, os

from tqdm.auto import tqdm

from xcai.misc import BEIR_DATASETS
from xclib.utils.sparse import retain_topk

def multiply_matrix(a:sp.csr_matrix, b:sp.csr_matrix, batch_size=1000):
    output = []
    for i in tqdm(range(0, a.shape[0], batch_size)):
        o = a[i: i+batch_size] @ b
        output.append(o)
    return sp.vstack(output)


if __name__ == "__main__":

    data_beir_dir = (
        f"/data/outputs/upma/16_beir-gpt-intent-substring-query-linker-with-ngame-loss-001/cross_predictions/document-intent-substring_simple/"
    )

    beir_msmarco_dir = (
        f"/data/outputs/upma/07_msmarco-gpt-intent-substring-linker-with-ngame-loss-002/predictions/document-intent-substring-simple-label-intent/"
    )

    save_dir = (
        "/data/outputs/upma/07_msmarco-gpt-intent-substring-linker-with-ngame-loss-002/cross_predictions/"
        "from_document-intent-substring-simple-label-intent_to_intent/using_16_beir-gpt-intent-substring-query-linker-with-ngame-loss-001/" 
    )
    os.makedirs(save_dir, exist_ok=True)

    for dataset in tqdm(BEIR_DATASETS):
        print(dataset)

        data_beir_file = f"{data_beir_dir}/test_predictions_{dataset}.npz"
        beir_msmarco_file = f"{beir_msmarco_dir}/document-intent-substring-simple-label-intent_predictions_{dataset}.npz"

        data_beir, beir_msmarco = sp.load_npz(data_beir_file), sp.load_npz(beir_msmarco_file)

        data_beir = retain_topk(data_beir, k=5)
        beir_msmarco = retain_topk(beir_msmarco, k=1)

        data_msmarco = multiply_matrix(data_beir, beir_msmarco, batch_size=1000)
        sp.save_npz(f"{save_dir}/test_predictions_{dataset}.npz", data_msmarco)

