#!/usr/bin/env python
"""
Standalone script to evaluate text embedding models (e.g. Nomic Embed, KaLM-Embedding,
or other custom SentenceTransformer models) on BEIR benchmark datasets.
Downloads datasets using the BEIR library, loads the model, formats the prompts,
runs exact search, and prints evaluation metrics (NDCG@10), reporting the average score.
"""

import os
import argparse
import logging
from typing import List, Dict
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.datasets.data_loader_hf import HFDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Standard BEIR datasets
STANDARD_BEIR_DATASETS = [
    "scifact",
    "nfcorpus",
    "fiqa",
    "arguana",
    "scidocs",
    "quora",
    "webis-touche2020",
    "trec-covid",
    "climate-fever",
    "fever",
    "hotpotqa",
    "nq",
    "dbpedia-entity",
]


class UnifiedBEIRModel:
    """
    Wrapper for SentenceTransformer models to comply with BEIR's DenseRetrievalExactSearch interface.
    Dynamically configures prefixes for query and document inputs based on the selected model type.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "auto",
        query_prefix: str = None,
        doc_prefix: str = None,
        device: str = None,
    ):
        logger.info(f"Loading SentenceTransformer model: {model_name} on device: {device}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.model_name = model_name.lower()

        # Auto-detect model type if not explicitly set
        if model_type == "auto":
            if "nomic" in self.model_name:
                self.model_type = "nomic"
            elif "kalm" in self.model_name:
                self.model_type = "kalm"
            else:
                self.model_type = "custom"
        else:
            self.model_type = model_type.lower()

        logger.info(f"Model type resolved to: {self.model_type}")

        # Set default prefixes based on model type
        if self.model_type == "nomic":
            self.query_prefix = "search_query: " if query_prefix is None else query_prefix
            self.doc_prefix = "search_document: " if doc_prefix is None else doc_prefix
        elif self.model_type == "kalm":
            self.query_prefix = (
                "Instruct: Given a query, retrieve documents that answer the query \n Query: "
                if query_prefix is None
                else query_prefix
            )
            self.doc_prefix = "" if doc_prefix is None else doc_prefix
        else:
            self.query_prefix = "" if query_prefix is None else query_prefix
            self.doc_prefix = "" if doc_prefix is None else doc_prefix

        logger.info(f"Using Query Prefix: {repr(self.query_prefix)}")
        logger.info(f"Using Document Prefix: {repr(self.doc_prefix)}")

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        logger.info(f"Encoding {len(queries)} queries with batch size {batch_size}...")
        prefixed_queries = [f"{self.query_prefix}{q}" for q in queries]
        return self.model.encode(
            prefixed_queries,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            **kwargs
        )

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 16, **kwargs) -> np.ndarray:
        logger.info(f"Encoding {len(corpus)} documents with batch size {batch_size}...")
        docs = []
        for doc in corpus:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            doc_text = f"{title} {text}".strip()
            docs.append(f"{self.doc_prefix}{doc_text}")

        return self.model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            **kwargs
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate text embedding models on BEIR datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact"],
        help="List of BEIR datasets to evaluate. Use 'all' for all standard BEIR datasets.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nomic-ai/nomic-embed-text-v1",
        help="HuggingFace model repository name (e.g. 'nomic-ai/nomic-embed-text-v1' or 'HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2.5').",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "nomic", "kalm", "custom"],
        help="Model type to apply default prompts (nomic, kalm, custom). 'auto' resolves based on model name.",
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
        help="Directory to save downloaded datasets and metrics.",
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
        datasets_to_run = STANDARD_BEIR_DATASETS
    logger.info(f"Datasets selected for evaluation: {datasets_to_run}")

    # Initialize model
    model = UnifiedBEIRModel(
        model_name=args.model_name,
        model_type=args.model_type,
        query_prefix=args.query_prefix,
        doc_prefix=args.doc_prefix,
        device=device,
    )

    # Dictionary to keep track of results
    ndcg_scores = {}

    os.makedirs(args.output_dir, exist_ok=True)
    datasets_dir = os.path.join(args.output_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    for dataset in datasets_to_run:
        logger.info(f"\n{'='*20} Evaluating {dataset} {'='*20}")
        try:
            # 1. Load dataset directly from Hugging Face via HFDataLoader
            logger.info(f"Loading dataset {dataset} directly from Hugging Face via BEIR HFDataLoader...")
            hf_repo = f"BeIR/{dataset}"
            corpus, queries, qrels = HFDataLoader(
                hf_repo=hf_repo,
                streaming=False,
                keep_in_memory=True
            ).load(split="test")
            logger.info(f"Loaded {len(queries)} queries and {len(corpus)} corpus documents.")

            # Get batch size for this dataset
            dataset_batch_size = batch_size_map.get(dataset, default_batch_size)

            # 3. Initialize BEIR dense retrieval exact search wrapper
            model_wrapper = DenseRetrievalExactSearch(model, batch_size=dataset_batch_size)
            retriever = EvaluateRetrieval(model_wrapper, score_function="cos_sim")

            # 4. Perform retrieval
            logger.info("Starting dense retrieval...")
            results = retriever.retrieve(corpus, queries)

            # 5. Evaluate and compute metrics
            logger.info("Computing metrics...")
            ndcg, _map, recall, precision = retriever.evaluate(
                qrels, retriever.k_values, retriever.score_function
            )

            # Record NDCG@10
            ndcg_10 = ndcg.get("NDCG@10", 0.0)
            ndcg_scores[dataset] = ndcg_10
            logger.info(f"Results for {dataset}: NDCG@10 = {ndcg_10:.5f}")
            for k in [1, 3, 5, 10]:
                logger.info(f"NDCG@{k}: {ndcg.get(f'NDCG@{k}', 0.0):.5f}")
                logger.info(f"Recall@{k}: {recall.get(f'Recall@{k}', 0.0):.5f}")

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


if __name__ == "__main__":
    main()
