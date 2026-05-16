import os, torch ,json, joblib, numpy as np, scipy.sparse as sp, argparse

from typing import Optional, Union, Callable, List
from tqdm.auto import tqdm

from xcai.basics import *
from xcai.models.memtr import MEM001, MEMConfig

def load_memtr_block(dataset:str, config_file:str, input_args:argparse.ArgumentParser, n_data_lnk_samples:Optional[int]=5, 
                    n_lbl_lnk_samples:Optional[int]=5, n_neg_lnk_samples:Optional[int]=5, data_lnk_topk:Optional[int]=5, 
                    lbl_lnk_topk:Optional[int]=5, neg_lnk_topk:Optional[int]=5, data_neg_topk:Optional[int]=None, 
                    train_label_topk:Optional[int]=None, num_label_samples:Optional[int]=1):
    config_key, fname = get_config_key(config_file)
    pkl_file = get_pkl_file(input_args.pickle_dir, f"{dataset}_{fname}_distilbert-base-uncased", input_args.use_sxc_sampler,
                            input_args.exact, input_args.only_test)

    train_data_meta_topk = {"lnk_meta": data_lnk_topk}
    if data_neg_topk is not None: train_data_meta_topk.update({"neg_meta": data_neg_topk})

    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    block = build_block(pkl_file, config_file, input_args.use_sxc_sampler, config_key, do_build=input_args.build_block, 
                        only_test=input_args.only_test, main_oversample=True, meta_oversample=True, return_scores=False, 
                        n_slbl_samples=num_label_samples, train_label_topk=train_label_topk,

                        n_sdata_meta_samples={"lnk_meta": n_data_lnk_samples, "neg_meta": 1},
                        n_slbl_meta_samples={"lnk_meta": n_lbl_lnk_samples, "neg_meta": 1},
                        n_sneg_meta_samples={"lnk_meta": n_neg_lnk_samples, "neg_meta": 1},
                        
                        train_data_meta_topk=train_data_meta_topk, test_data_meta_topk={"lnk_meta": data_lnk_topk}, 
                        train_label_meta_topk={"lnk_meta": lbl_lnk_topk}, test_label_meta_topk={"lnk_meta": lbl_lnk_topk},
                        train_neg_meta_topk={"lnk_meta": neg_lnk_topk}, test_neg_meta_topk={"lnk_meta": neg_lnk_topk}, 
                        ignore_data_info=True, ignore_lbl_info=True)
    
    train_dset, test_dset = None if block.train is None else block.train.dset, block.test.dset

    return train_dset, test_dset
    
def memtr_run(output_dir:str, input_args:argparse.ArgumentParser, mname:str, test_dset:Union[XCDataset, SXCDataset],
              train_embedding_file:str, test_embedding_file:str, label_embedding_file:str,
              train_dset:Optional[Union[XCDataset, SXCDataset]]=None, collator:Optional[Callable]=identity_collate_fn, 

              test_identifiers:Optional[List]=None, label_identifiers:Optional[List]=None,

              train_batch_size:Optional[int]=128, eval_batch_size:Optional[int]=400, save_dir_name:Optional[str]=None,
              num_input_metadata:Optional[int]=5, use_calib_loss:Optional[bool]=False, calib_loss_weight:Optional[float]=0.1, 

              update_config_during_inference:Optional[bool]=False, resume_from_checkpoint:Optional[bool]=None, 
              use_saved_representation_for_indexing:Optional[bool]=False, prefix_for_saved_representation_for_indexing:Optional[str]=None, 

              dataset:Optional[str]=None, load_model_type:Optional[str]="best"):

    if dataset is not None and dataset in set(["dbpedia-entity", "trec-covid", "webis-touche2020", "nfcorpus", "trecdl19", "trecdl20"]):
        label_names += ["plbl2data_scores"]
    
    label_names = ["lnk2data_input_ids", "lnk2data_attention_mask", "lnk2data_idx", "lnk2data_data2ptr"]
    
    args = XCLearningArguments(
        output_dir=output_dir,
        logging_first_step=True,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        representation_num_beams=200,
        representation_accumulation_steps=10,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        save_total_limit=5,
        num_train_epochs=50,
        predict_with_representation=True,
        representation_search_type='BRUTEFORCE',
        search_normalize=normalize,

        adam_epsilon=1e-6,
        warmup_steps=1000,
        weight_decay=0.01,
        learning_rate=6e-5,
        label_names=label_names,

        group_by_cluster=True,
        use_data_metadata_for_clustering=True,
        num_clustering_warmup_epochs=10,
        num_cluster_update_epochs=5,
        num_cluster_size_update_epochs=25,
        clustering_type='EXPO',
        minimum_cluster_size=2,
        maximum_cluster_size=1600,

        data_aug_meta_name="lnk",
        use_label_metadata=False,

        metric_for_best_model='N@10',
        load_best_model_at_end=True,
        target_indices_key='plbl2data_idx',
        target_pointer_key='plbl2data_data2ptr',

        use_encoder_parallel=True,
        max_grad_norm=None,
        fp16=True,

        use_cpu_for_searching=True,
        use_cpu_for_clustering=True,

        prefix_for_saved_representation_for_indexing=prefix_for_saved_representation_for_indexing,
        use_saved_representation_for_indexing=use_saved_representation_for_indexing,
    )

    config = MEMConfig(
        num_train_data=train_dset.n_data,
        num_test_data=test_dset.n_data,
        num_labels=train_dset.n_lbl,
        num_metadata=train_dset.meta["lnk_meta"].n_meta,
        data_aug_meta_prefix="lnk2data",
        base_model_dim=12,
        
        num_negatives=10,
        tau=0.1,
        reduction="mean",
    
        use_encoder_parallel=True,
        loss_function="ranking",
    )

    trn_embeds = torch.load(train_embedding_file)
    tst_embeds = torch.load(test_embedding_file)
    lbl_embeds = torch.load(label_embedding_file)

    def model_fn(mname:Optional[str]=None):
        model = MEM001.from_pretrained('distilbert-base-uncased', config=config)
        model.init_encoder_embeddings(trn_embeds, tst_embeds, lbl_embeds)
        return model

    metric = BeirMetric(test_dset.n_lbl, k_values=[1, 3, 5, 10], qry_ids=test_identifiers, lbl_ids=label_identifiers)

    model = load_model(args.output_dir, model_fn, do_inference=check_inference_mode(input_args), use_pretrained=input_args.use_pretrained, 
                       update_config_during_inference=update_config_during_inference, config=config, type=load_model_type)
        
    learn = XCLearner(
        model=model,
        args=args,
        train_dataset=train_dset,
        eval_dataset=test_dset,
        data_collator=collator,
        compute_metrics=metric,
    )

    return main(learn, input_args, n_lbl=test_dset.n_lbl, save_dir_name=save_dir_name, 
                resume_from_checkpoint=resume_from_checkpoint)
    

if __name__ == '__main__':
    input_args = parse_args()

    output_dir = "/data/suchith/outputs/upma/23_memtr-with-nvembed-encoder-and-distilbert-memory-module-001"

    input_args.use_sxc_sampler = True
    input_args.pickle_dir = "/data/suchith/datasets/processed/"
    mname = "distilbert-base-uncased"

    config_file = "configs/msmarco/hipporag/data_lbl_hipporag_nvembed-negatives-topk-05.json"

    train_dset, test_dset = load_memtr_block("msmarco", config_file, input_args)

    memtr_run(output_dir, input_args, mname, test_dset, train_dset=train_dset, train_batch_size=128)

