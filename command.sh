# Conflation code
python scripts/01_conflation.py --pred_file /data/datasets/msmarco/XC/entity-gpt_ngame_trn_X_Y.npz --lbl_file //data/datasets/msmarco/XC/raw_data/entity_gpt.raw.txt  --trn_file /data/datasets/msmarco/XC/entity_gpt_trn_X_Y.npz --tst_file /data/datasets/msmarco/XC/entity_gpt_tst_X_Y.npz --topk 3 --batch_size 1024 --min_thresh 2 --max_thresh 100 --freq_thresh 50 --score_thresh 25 --diff_thresh 0.1 --embed_file /data/outputs/mogicX/01-msmarco-gpt-entity-linker-001/predictions/label_repr_full.pth

