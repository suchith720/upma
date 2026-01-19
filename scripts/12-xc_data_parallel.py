import torch, torch.nn as nn
from transformers import BatchEncoding
from transformers import AutoTokenizer

from xcai.main import *
from xcai.learner import XCDataParallel

class MyModel(nn.Module):

    def forward(self, **kwargs):
        for k,v in kwargs.items(): 
            if isinstance(v, torch.Tensor) and "input_ids" in k: 
                print(k, ': ', v, ', ', v.device)
        return kwargs

if __name__ == "__main__":
    pkl_file = "/home/aiscuser/wikiseealsotitles.joblib"
    data_dir = "/data/datasets/benchmarks/"

    block = build_block(pkl_file, "wikiseealsotitles", True, "data_meta", data_dir=data_dir, n_slbl_samples=2, main_oversample=False, 
                        n_sdata_meta_samples=3, n_slbl_meta_samples=2)

    batch = BatchEncoding(block.train.dset.__getitems__([10, 15, 20, 30]))
    batch = batch.to('cuda')

    model = XCDataParallel(module=MyModel())
    o = model(**batch)

