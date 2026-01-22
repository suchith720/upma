import joblib, os, logging, traceback, torch, scipy.sparse as sp, pandas as pd

from datetime import datetime
from tqdm.auto import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset

from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import LambdaLoss, NDCGLoss2PPScheme
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.util import mine_hard_negatives

from xcai.basics import *

def main():
    input_args = parse_args()
    output_dir = "/home/aiscuser/scratch1/outputs/upma/18_cross-encoder-training-ms-marco-lambda-hard-neg-002"
    pickle_dir = "/home/aiscuser/scratch1/datasets/processed/"

    model_name = "microsoft/MiniLM-L12-H384-uncased"

    # Set the log level to INFO to get more information
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # train_batch_size and eval_batch_size inform the size of the batches, while mini_batch_size is used by the loss
    # to subdivide the batch into smaller parts. This mini_batch_size largely informs the training speed and memory usage.
    # Keep in mind that the loss does not process `train_batch_size` pairs, but `train_batch_size * num_docs` pairs.
    train_batch_size = 16
    eval_batch_size = 16
    mini_batch_size = 16
    num_epochs = 1
    max_docs = None

    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 1. Define our CrossEncoder model
    # Set the seed so the new classifier weights are identical in subsequent runs
    torch.manual_seed(12)
    model = CrossEncoder(model_name, num_labels=1)
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/microsoft/ms_marco
    input_args.use_sxc_sampler = True
    input_args.pickle_dir = "/home/aiscuser/scratch1/datasets/processed/"

    config_file = "configs/msmarco/intent_substring/data_lbl_ngame-gpt-intent-substring-conflation-01_ce-negatives-topk-05-linker-label-intent-substring_exact.json"
    train_dset, test_dset = load_upma_block("msmarco", config_file, input_args)

    def get_positive_dataset(dset):
        data_info, lbl_info = dset.data.data_info["input_text"], dset.data.lbl_info["input_text"]

        queries, docs, labels = [], [], []
        for idx in tqdm(range(dset.data.data_lbl.shape[0])):
            indices = dset.data.data_lbl[idx].indices
            queries.append(data_info[idx]) 
            docs.append([lbl_info[i] for i in indices])
            labels.append([1] * len(indices))

        return Dataset.from_dict({"query": queries, "docs": docs, "labels": labels})

    pos_dataset = get_positive_dataset(train_dset)
    logging.info(f"Created {len(pos_dataset):_} query-positive pairs")

    def get_negative_dataset(dset):
        data_info, lbl_info = dset.data.data_info["input_text"], dset.meta["neg_meta"].meta_info["input_text"]

        queries, docs, labels = [], [], []
        for idx in tqdm(range(dset.meta["neg_meta"].data_meta.shape[0])):
            indices = dset.meta["neg_meta"].data_meta[idx].indices
            queries.append(data_info[idx]) 
            docs.append([lbl_info[i] for i in indices])
            labels.append([0] * len(indices))

        return Dataset.from_dict({"query": queries, "docs": docs, "labels": labels})

    neg_dataset = get_negative_dataset(train_dset)
    logging.info(f"Created {len(neg_dataset):_} query-negative pairs")

    # Concatenate the two datasets into one to  form training dataset
    train_dataset = concatenate_datasets([pos_dataset, neg_dataset])

    # Evaluation dataset

    def get_eval_dataset(dset, pred_file, lbl_file):
        pos_dataset = get_positive_dataset(dset)

        pred_lbl = sp.load_npz(pred_file)
        lbl_info = pd.read_csv(lbl_file)["text"]

        assert len(pos_dataset) == pred_lbl.shape[0]

        eval_dataset = []
        for idx in range(pred_lbl.shape[0]):
            o = pos_dataset[idx]

            indices = pred_lbl[idx].indices
            sort_idx = np.argsort(pred_lbl[idx].data)[::-1]
            documents = [lbl_info[i] for i in indices[sort_idx]]
            o.update({"documents": documents})

            eval_dataset.append(o)

        return eval_dataset

    pred_file = "/data/outputs/upma/09_upma-with-ngame-gpt-intent-substring-linker-for-msmarco-008/predictions/test_predictions_msmarco.npz"
    lbl_file = "/data/datasets/beir/msmarco/XC/raw_data/label.raw.csv"

    eval_dataset = get_eval_dataset(test_dset, pred_file, lbl_file)

    logging.info(train_dataset)

    # 3. Define our training loss
    loss = LambdaLoss(
        model=model,
        weighting_scheme=NDCGLoss2PPScheme(),
        mini_batch_size=mini_batch_size,
    )

    # 4. Define the evaluator. We use the CENanoBEIREvaluator, which is a light-weight evaluator for English reranking
    evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=eval_batch_size)
    evaluator(model)

    return

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-msmarco-{short_model_name}-lambdaloss-hard-neg"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"{output_dir}/{run_name}_{dt}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=250,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, useful to include these in the model card
    evaluator(model)

    # 8. Save the final model
    final_output_dir = f"{output_dir}/{run_name}_{dt}/final"
    model.save_pretrained(final_output_dir)

if __name__ == "__main__":
    main()

