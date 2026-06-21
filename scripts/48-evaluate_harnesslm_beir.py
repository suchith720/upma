#!/usr/bin/env python
"""
Standalone script to evaluate text embedding models on BEIR benchmark datasets
using a dual-tower model configuration (separate query and document models).
Supports slicing/truncating document embeddings if the query tower has lower dimensions.
"""

import os
import argparse
import logging
from typing import List, Dict
import numpy as np
import torch
import scipy.sparse as sp
import json

from sentence_transformers import SentenceTransformer
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.datasets.data_loader_hf import HFDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from tqdm.auto import tqdm

from sugar.core import load_raw_file

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Standard BEIR datasets
BEIR_DATASETS = [
    "arguana",
    "scidocs",
    "scifact",
    "webis-touche2020",
    "trec-covid",
    "cqadupstack/android",
    "cqadupstack/english",
    "cqadupstack/gaming",
    "cqadupstack/gis",
    "cqadupstack/mathematica",
    "cqadupstack/physics",
    "cqadupstack/programmers",
    "cqadupstack/stats",
    "cqadupstack/tex",
    "cqadupstack/unix",
    "cqadupstack/webmasters",
    "cqadupstack/wordpress",
    "fiqa",
    "quora",
    "msmarco",
    "climate-fever",
    "dbpedia-entity",
    "fever",
    "hotpotqa",
    "nfcorpus",
    "nq",
]

def collate_beir_metrics(metric_dir:str):
    beir_metrics = {}
    for dataset in BEIR_DATASETS:
        dataset = dataset.replace("/", "-")
        
        fname = f"{metric_dir}/{dataset}.json"
        if os.path.exists(fname):
            with open(fname) as file:
                beir_metrics.update(json.load(file))
            
    with open(f"{metric_dir}/beir.json", "w") as file:
        json.dump(beir_metrics, file, indent=4)


class DataParallelEncoder(torch.nn.Module):
    def __init__(self, st_model):
        super().__init__()
        self.st_model = st_model

    def forward(self, **features):
        return self.st_model(features)["sentence_embedding"]


class UnifiedBEIRModel:
    """
    Wrapper for dual SentenceTransformer models (two-tower) to comply with BEIR's DenseRetrievalExactSearch interface.
    """

    def __init__(
        self,
        query_model_name: str,
        doc_model_name: str,
        query_model_type: str = "auto",
        doc_model_type: str = "auto",
        query_prefix: str = None,
        doc_prefix: str = None,
        device: str = None,
        use_data_parallel: bool = False,
    ):
        logger.info(f"Loading query model: {query_model_name} and doc model: {doc_model_name} on device: {device}")
        
        self.query_model_name = query_model_name.lower()
        self.doc_model_name = doc_model_name.lower()
        self.device = device
        self.use_data_parallel = use_data_parallel

        # Prepare kwargs for query model
        query_st_kwargs = {"trust_remote_code": True, "device": device}
        query_model_kwargs = {}
        query_tokenizer_kwargs = {}
        if "qwen" in self.query_model_name:
            query_tokenizer_kwargs["padding_side"] = "left"
            if "8b" in self.query_model_name:
                if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    query_model_kwargs["torch_dtype"] = torch.bfloat16
                elif device == "cuda" and torch.cuda.is_available():
                    query_model_kwargs["torch_dtype"] = torch.float16
                elif device in ("mps", "cuda"):
                    query_model_kwargs["torch_dtype"] = torch.float16

        if query_model_kwargs:
            query_st_kwargs["model_kwargs"] = query_model_kwargs
        if query_tokenizer_kwargs:
            query_st_kwargs["tokenizer_kwargs"] = query_tokenizer_kwargs

        # Prepare kwargs for doc model
        doc_st_kwargs = {"trust_remote_code": True, "device": device}
        doc_model_kwargs = {}
        doc_tokenizer_kwargs = {}
        if "qwen" in self.doc_model_name:
            doc_tokenizer_kwargs["padding_side"] = "left"
            if "8b" in self.doc_model_name:
                if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    doc_model_kwargs["torch_dtype"] = torch.bfloat16
                elif device == "cuda" and torch.cuda.is_available():
                    doc_model_kwargs["torch_dtype"] = torch.float16
                elif device in ("mps", "cuda"):
                    doc_model_kwargs["torch_dtype"] = torch.float16

        if doc_model_kwargs:
            doc_st_kwargs["model_kwargs"] = doc_model_kwargs
        if doc_tokenizer_kwargs:
            doc_st_kwargs["tokenizer_kwargs"] = doc_tokenizer_kwargs

        # Load models
        logger.info(f"Loading query SentenceTransformer with kwargs: {query_st_kwargs}")
        self.query_model = SentenceTransformer(query_model_name, **query_st_kwargs)
        self.query_model.max_seq_length = 512

        logger.info(f"Loading doc SentenceTransformer with kwargs: {doc_st_kwargs}")
        self.doc_model = SentenceTransformer(doc_model_name, **doc_st_kwargs)
        self.doc_model.max_seq_length = 512

        # Data Parallelism
        if self.use_data_parallel and torch.cuda.device_count() > 1:
            logger.info(f"Initializing torch.nn.DataParallel across {torch.cuda.device_count()} GPUs...")
            self.parallel_query_model = torch.nn.DataParallel(DataParallelEncoder(self.query_model))
            self.parallel_query_model.to(device)
            self.parallel_doc_model = torch.nn.DataParallel(DataParallelEncoder(self.doc_model))
            self.parallel_doc_model.to(device)
        else:
            self.use_data_parallel = False
            self.parallel_query_model = None
            self.parallel_doc_model = None

        # Resolve query model type
        if query_model_type == "auto":
            if "nomic" in self.query_model_name:
                self.query_model_type = "nomic"
            elif "kalm" in self.query_model_name:
                self.query_model_type = "kalm"
            elif "qwen" in self.query_model_name:
                self.query_model_type = "qwen"
            else:
                self.query_model_type = "custom"
        else:
            self.query_model_type = query_model_type.lower()

        # Resolve doc model type
        if doc_model_type == "auto":
            if "nomic" in self.doc_model_name:
                self.doc_model_type = "nomic"
            elif "kalm" in self.doc_model_name:
                self.doc_model_type = "kalm"
            elif "qwen" in self.doc_model_name:
                self.doc_model_type = "qwen"
            else:
                self.doc_model_type = "custom"
        else:
            self.doc_model_type = doc_model_type.lower()

        logger.info(f"Query model type resolved to: {self.query_model_type}")
        logger.info(f"Doc model type resolved to: {self.doc_model_type}")

        # Set default query prefixes
        if self.query_model_type == "nomic":
            self.query_prefix = "search_query: " if query_prefix is None else query_prefix
        elif self.query_model_type == "kalm":
            self.query_prefix = (
                "Instruct: Given a query, retrieve documents that answer the query \n Query: "
                if query_prefix is None
                else query_prefix
            )
        elif self.query_model_type == "qwen":
            self.query_prefix = (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
                if query_prefix is None
                else query_prefix
            )
        else:
            self.query_prefix = "" if query_prefix is None else query_prefix

        # Set default doc prefixes
        if self.doc_model_type == "nomic":
            self.doc_prefix = "search_document: " if doc_prefix is None else doc_prefix
        else:
            self.doc_prefix = "" if doc_prefix is None else doc_prefix

        logger.info(f"Using Query Prefix: {repr(self.query_prefix)}")
        logger.info(f"Using Document Prefix: {repr(self.doc_prefix)}")

    def _encode_texts(self, model: SentenceTransformer, parallel_model, texts: List[str], batch_size: int = 256) -> np.ndarray:
        if self.use_data_parallel and parallel_model is not None:
            logger.info("Encoding using PyTorch DataParallel...")
            embeddings = []
            features = model.tokenize(texts)
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_features = {
                    k: v[i : i + batch_size].to(self.device) if isinstance(v, torch.Tensor) else v[i : i + batch_size]
                    for k, v in features.items()
                }
                # Inject modality configurations for newer/custom sentence-transformers versions
                batch_features["modality"] = "text"
                batch_features["modality_name"] = "text"
                with torch.no_grad():
                    batch_embeddings = parallel_model(**batch_features)
                    embeddings.append(batch_embeddings.cpu().numpy())
            return np.concatenate(embeddings, axis=0)
        else:
            return model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        logger.info(f"Encoding {len(queries)} queries with batch size {batch_size}...")
        prefixed_queries = [f"{self.query_prefix}{q}" for q in queries]
        return self._encode_texts(self.query_model, self.parallel_query_model, prefixed_queries, batch_size=batch_size)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 16, **kwargs) -> np.ndarray:
        logger.info(f"Encoding {len(corpus)} documents with batch size {batch_size}...")
        docs = []
        for doc in corpus:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            doc_text = f"{title} {text}".strip()
            docs.append(f"{self.doc_prefix}{doc_text}")
        
        doc_embeddings = self._encode_texts(self.doc_model, self.parallel_doc_model, docs, batch_size=batch_size)

        # Slice the doc embeddings if the query tower dimension is lower
        query_dim = self.query_model.get_sentence_embedding_dimension()
        doc_dim = doc_embeddings.shape[1]
        if doc_dim > query_dim:
            logger.info(f"Truncating document embeddings dimension from {doc_dim} to match query dimension {query_dim}")
            doc_embeddings = doc_embeddings[:, :query_dim]

        return doc_embeddings

def main():
    parser = argparse.ArgumentParser(description="Evaluate text embedding models on BEIR datasets using dual towers.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact"],
        help="List of BEIR datasets to evaluate. Use 'all' for all standard BEIR datasets.",
    )
    parser.add_argument(
        "--query_model_name",
        type=str,
        default="nomic-ai/nomic-embed-text-v1",
        help="HuggingFace model repository name for the query tower.",
    )
    parser.add_argument(
        "--doc_model_name",
        type=str,
        default="nomic-ai/nomic-embed-text-v1",
        help="HuggingFace model repository name for the document tower.",
    )
    parser.add_argument(
        "--query_model_type",
        type=str,
        default="auto",
        choices=["auto", "nomic", "kalm", "qwen", "custom"],
        help="Query model type to apply default prompts (nomic, kalm, qwen, custom). 'auto' resolves based on model name.",
    )
    parser.add_argument(
        "--doc_model_type",
        type=str,
        default="auto",
        choices=["auto", "nomic", "kalm", "qwen", "custom"],
        help="Document model type to apply default prompts (nomic, kalm, qwen, custom). 'auto' resolves based on model name.",
    )
    parser.add_argument(
        "--query_prefix",
        type=str,
        default=None,
        help="Override/provide query prefix instruction.",
    )
    parser.add_argument(
        "--doc_prefix",
        type=str,
        default=None,
        help="Override/provide document prefix instruction.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./beir_evaluation",
        help="Directory to save downloaded datasets.",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="256",
        help="Batch size for model encoding. Can be an integer or a dictionary mapping dataset names to integer sizes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for encoding (e.g. 'cuda', 'mps', 'cpu'). If None, automatically detects.",
    )
    parser.add_argument(
        "--use_data_parallel",
        action="store_true",
        help="Use PyTorch DataParallel across multiple GPUs for faster encoding.",
    )
    parser.add_argument(
        "--metric_dir",
        type=str,
        default="./beir_evaluation/metrics",
        help="Directory to save metrics.",
    )

    args = parser.parse_args()

    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Determine batch size mapping
    try:
        default_batch_size = int(args.batch_size)
        batch_size_map = {}
    except ValueError:
        import ast
        batch_size_map = ast.literal_eval(args.batch_size)
        default_batch_size = batch_size_map.get("default", 256)

    # Determine datasets to evaluate
    datasets_to_run = args.datasets
    if len(datasets_to_run) == 1 and datasets_to_run[0].lower() == "all":
        datasets_to_run = BEIR_DATASETS
    logger.info(f"Datasets selected for evaluation: {datasets_to_run}")

    # Initialize model
    model = UnifiedBEIRModel(
        query_model_name=args.query_model_name,
        doc_model_name=args.doc_model_name,
        query_model_type=args.query_model_type,
        doc_model_type=args.doc_model_type,
        query_prefix=args.query_prefix,
        doc_prefix=args.doc_prefix,
        device=device,
        use_data_parallel=args.use_data_parallel,
    )

    # Dictionary to keep track of results
    ndcg_scores = {}

    os.makedirs(args.output_dir, exist_ok=True)
    datasets_dir = os.path.join(args.output_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    metric_dir = f"{args.metric_dir}/metrics"
    os.makedirs(metric_dir, exist_ok=True)

    pred_dir = f"{args.metric_dir}/predictions"
    os.makedirs(pred_dir, exist_ok=True)
                
    for dataset in datasets_to_run:
        data_dir = f"/data/datasets/beir/{dataset}/XC/"

        if model.query_model_type == "qwen":
            try:
                from xcai.maggi.utils import DATASETS, get_instruction
                instruction = "/home/sasokan/suchith/xcai/xcai/models/nvembed/instructions.json"
                instruction = get_instruction(instruction, DATASETS[dataset])["query"]
                model.query_prefix = f"Instruct: {instruction} \n Query: "
            except Exception as e:
                logger.warning(f"Could not load custom Qwen instruction: {e}. Using default prefix.")

        logger.info(f"\n{'='*20} Evaluating {dataset} {'='*20}")
        try:
            # 1. Load dataset directly from Hugging Face or Local Files
            if "cqadupstack" in dataset or "msmarco" in dataset:
                # Handle cqadupstack sub-datasets (e.g. cqadupstack/android)
                sub_dataset = dataset.split("/")[-1]
                logger.info(f"Loading cqadupstack sub-dataset '{sub_dataset}' directly from local blob storage...")

                data_ids, data_txt = load_raw_file(f"{data_dir}/raw_data/test.raw.csv")
                queries = [{"id":i, "text":t} for i,t in zip(data_ids, data_txt)]

                lbl_ids, lbl_txt = load_raw_file(f"{data_dir}/raw_data/label.raw.csv")
                corpus = [{"id":i, "text":t} for i,t in zip(lbl_ids, lbl_txt)]

                data_lbl = sp.load_npz(f"{data_dir}/tst_X_Y.npz")

                qrels = list()
                for i,row in enumerate(data_lbl):
                    for j,sc in zip(row.indices, row.data):
                        qrels.append({"query_id":data_ids[i], "corpus_id":lbl_ids[j], "score": sc})

            else:
                # Standard BEIR datasets load via HFDataLoader
                logger.info(f"Loading dataset {dataset} directly from Hugging Face via BEIR HFDataLoader...")
                hf_repo = f"BeIR/{dataset}"
                corpus, queries, qrels = HFDataLoader(
                    hf_repo=hf_repo,
                    streaming=False,
                    keep_in_memory=True
                ).load(split="test")

            # Convert queries to BEIR dictionary format if it is a Hugging Face Dataset
            if not isinstance(queries, dict):
                queries_dict = {}
                for item in queries:
                    qid = str(item.get("id", item.get("_id")))
                    queries_dict[qid] = item.get("text", "")
                queries = queries_dict

            # Convert corpus to BEIR dictionary format if it is a Hugging Face Dataset
            if not isinstance(corpus, dict):
                corpus_dict = {}
                for item in corpus:
                    doc_id = str(item.get("id", item.get("_id")))
                    corpus_dict[doc_id] = {
                        "title": item.get("title", ""),
                        "text": item.get("text", "")
                    }
                corpus = corpus_dict

            # Convert qrels to BEIR dictionary format if it is a Hugging Face Dataset
            if not isinstance(qrels, dict):
                qrels_dict = {}
                for item in qrels:
                    qid = str(item.get("query-id", item.get("query_id")))
                    did = str(item.get("corpus-id", item.get("corpus_id")))
                    score = int(item.get("score", 1))
                    qrels_dict.setdefault(qid, {})[did] = score
                qrels = qrels_dict

            logger.info(f"Loaded {len(queries)} queries and {len(corpus)} corpus documents.")

            # Get batch size for this dataset
            dataset_batch_size = batch_size_map.get(dataset, default_batch_size)

            # 3. Initialize BEIR dense retrieval exact search wrapper
            model_wrapper = DenseRetrievalExactSearch(model, batch_size=dataset_batch_size, 
                                                      corpus_chunk_size=500_000, show_progress_bar=True)
            retriever = EvaluateRetrieval(model_wrapper, score_function="cos_sim")

            # 4. Perform retrieval
            logger.info("Starting dense retrieval...")
            results = retriever.retrieve(corpus, queries)

            dataset_prefix = dataset.replace("/", "-")
                
            data_ids, data_txt = load_raw_file(f"{data_dir}/raw_data/test.raw.csv")
            lbl_ids, lbl_txt = load_raw_file(f"{data_dir}/raw_data/label.raw.csv")
            lbl_ids2idx = {str(i):idx for idx,i in enumerate(lbl_ids)}
            data, indices, indptr = [], [], [0]
            for i in data_ids:
                for l, sc in results[str(i)].items():
                    if l in lbl_ids2idx:
                        data.append(sc)
                        indices.append(lbl_ids2idx[l])
                    else:
                        print(f"Invalid label predicted: {l}")
                indptr.append(len(data))
            pred_lbl = sp.csr_matrix((data, indices, indptr), dtype=np.float32, shape=(len(data_ids), len(lbl_ids)))
            sp.save_npz(f"{pred_dir}/test_predictions_{dataset_prefix}.npz", pred_lbl)

            # 5. Evaluate and compute metrics
            logger.info("Computing metrics...")
            ndcg, _map, recall, precision = retriever.evaluate(
                qrels, results, retriever.k_values, 
            )
            mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

            metrics = {}
            for name, vals in [("NDCG", ndcg), ("Recall", recall), ("Precision", precision), ("MAP", _map), ("MRR", mrr)]:
                for k, v in vals.items(): metrics.update({f"{name}@{k.split('@')[1]}": v})

            with open(f"{metric_dir}/{dataset_prefix}.json", "w") as file:
                json.dump({dataset: metrics}, file, indent=4)
            ndcg_scores[dataset_prefix] = metrics["NDCG@10"]

        except Exception as e:
            logger.error(f"Failed to evaluate {dataset}: {str(e)}", exc_info=True)

    # Report final results
    logger.info(f"\n{'='*20} Summary of Evaluation {'='*20}")
    if ndcg_scores:
        for dataset, score in ndcg_scores.items():
            print(f"{dataset}: NDCG@10 = {score:.5f}")

        avg_ndcg = sum(ndcg_scores.values()) / len(ndcg_scores)
        print(f"\nAverage NDCG@10: {avg_ndcg:.5f}")
    else:
        logger.warning("No datasets were successfully evaluated.")

    collate_beir_metrics(metric_dir)


if __name__ == "__main__":
    main()
