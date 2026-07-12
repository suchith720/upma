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
import scipy.sparse as sp
import json

from sentence_transformers import SentenceTransformer
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.datasets.data_loader_hf import HFDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch, HNSWFaissSearch
from tqdm.auto import tqdm

from xcai.misc import BEIR_DATASETS
from sugar.core import load_raw_file
from xclib.utils.sparse import retain_topk

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class HNSWSQFaissSearchCompact(HNSWFaissSearch):
    """Newer BEIR marks encode/search_from_files abstract but never uses them
    in the retrieve()->search() path. Stub them so the class is instantiable.

    NOTE: uses HNSWFaissSearch (not HNSWSQFaissSearch): the SQ variant in this
    BEIR build creates IndexHNSWSQ(dim+1) but trains via FaissTrainIndex on raw
    dim-D embeddings (no aux-dim augmentation) -> `assert d == self.d` crash.
    HNSWFaissSearch augments both corpus and queries correctly."""

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def search_from_files(self, *args, **kwargs):
        raise NotImplementedError

# Standard BEIR datasets
DATASETS = [
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
        use_data_parallel: bool = False,
    ):
        logger.info(f"Loading SentenceTransformer model: {model_name} on device: {device}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.model.max_seq_length = 512
        self.model_name = model_name.lower()
        self.device = device
        self.use_data_parallel = use_data_parallel

        if self.use_data_parallel and torch.cuda.device_count() > 1:
            logger.info(f"Initializing torch.nn.DataParallel across {torch.cuda.device_count()} GPUs...")
            self.parallel_model = torch.nn.DataParallel(DataParallelEncoder(self.model))
            self.parallel_model.to(device)
        else:
            self.use_data_parallel = False
            self.parallel_model = None

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

        # Caching corpus embeddings attributes
        self.current_dataset = None
        self.embeddings_dir = None
        self.current_doc_cursor = 0
        self.total_corpus_size = 0
        self.encoded_slices = []
        self.entire_corpus_embeddings = None
        self.cache_path = None

    def set_dataset(self, dataset_name: str, corpus: Dict[str, Dict[str, str]], embeddings_dir: str):
        self.current_dataset = dataset_name
        self.embeddings_dir = embeddings_dir
        self.current_doc_cursor = 0
        self.total_corpus_size = len(corpus)
        self.encoded_slices = []

        # Ensure directory exists
        os.makedirs(embeddings_dir, exist_ok=True)

        # Sanitise model name for filename
        sanitized_model_name = self.model_name.strip("/").replace("/", "_").replace("\\", "_")
        cache_filename = f"{dataset_name.replace('/', '-')}_{sanitized_model_name}_embeddings.npy"
        self.cache_path = os.path.join(embeddings_dir, cache_filename)

        if os.path.exists(self.cache_path):
            logger.info(f"Loading cached corpus embeddings from {self.cache_path}...")
            try:
                self.entire_corpus_embeddings = np.load(self.cache_path)
                if len(self.entire_corpus_embeddings) != self.total_corpus_size:
                    logger.warning(
                        f"Cached embeddings size ({len(self.entire_corpus_embeddings)}) "
                        f"does not match corpus size ({self.total_corpus_size}). Discarding cache."
                    )
                    self.entire_corpus_embeddings = None
            except Exception as e:
                logger.error(f"Error loading cached embeddings: {e}. Discarding cache.", exc_info=True)
                self.entire_corpus_embeddings = None
        else:
            logger.info(f"No cached embeddings found at {self.cache_path}. Will encode and save.")
            self.entire_corpus_embeddings = None

    def _encode_texts(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        if self.use_data_parallel:
            logger.info("Encoding using PyTorch DataParallel...")
            embeddings = []
            features = self.model.tokenize(texts)
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_features = {
                    k: v[i : i + batch_size].to(self.device) if isinstance(v, torch.Tensor) else v[i : i + batch_size]
                    for k, v in features.items()
                }
                # Inject modality configurations for newer/custom sentence-transformers versions
                batch_features["modality"] = "text"
                batch_features["modality_name"] = "text"
                with torch.no_grad():
                    batch_embeddings = self.parallel_model(**batch_features)
                    embeddings.append(batch_embeddings.cpu().numpy())
            return np.concatenate(embeddings, axis=0)
        else:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        logger.info(f"Encoding {len(queries)} queries with batch size {batch_size}...")
        prefixed_queries = [f"{self.query_prefix}{q}" for q in queries]
        return self._encode_texts(prefixed_queries, batch_size=batch_size)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 16, **kwargs) -> np.ndarray:
        if self.entire_corpus_embeddings is not None:
            n = len(corpus)
            logger.info(f"Loading {n} documents from cache (cursor: {self.current_doc_cursor})...")
            embeddings_slice = self.entire_corpus_embeddings[self.current_doc_cursor : self.current_doc_cursor + n]
            self.current_doc_cursor += n
            return embeddings_slice

        logger.info(f"Encoding {len(corpus)} documents with batch size {batch_size}...")
        docs = []
        for doc in corpus:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            doc_text = f"{title} {text}".strip()
            docs.append(f"{self.doc_prefix}{doc_text}")
        doc_embeddings = self._encode_texts(docs, batch_size=batch_size)

        if self.cache_path is not None and self.total_corpus_size > 0:
            self.encoded_slices.append(doc_embeddings)
            self.current_doc_cursor += len(corpus)
            if self.current_doc_cursor >= self.total_corpus_size:
                logger.info(f"All corpus documents encoded. Saving embeddings to {self.cache_path}...")
                entire_embeddings = np.concatenate(self.encoded_slices, axis=0)
                np.save(self.cache_path, entire_embeddings)
                self.entire_corpus_embeddings = entire_embeddings
                self.encoded_slices = []

        return doc_embeddings

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
    parser.add_argument(
        "--pred_suffix",
        type=str,
        default=None,
        help="Suffix of the prediction file.",
    )
    parser.add_argument(
        "--nomic_data",
        action="store_true",
        help="Compute predictions on nomic training data.",
    )
    parser.add_argument(
        "--use_anns",
        action="store_true",
        help="Use ANNS for approximate search.",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="./beir_evaluation/corpus_embeddings",
        help="Directory to save/load cached document (corpus) embeddings.",
    )

    args = parser.parse_args()

    WIKIPEDIA_DATASETS = {
        "climate-fever",
        "dbpedia-entity",
        "fever",
        "hotpotqa",
        "nq",
    }

    NOMIC_DATASETS = {
        # "fever": "fever_hn_mine",
        "hotpotqa": "hotpotqa_hn_mine_shuffled",
        "nq": "nq_cocondensor_hn_mine_reranked_min15",
        "msmarco": "msmarco_distillation_simlm_rescored_reranked_min15",
    }

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
    if args.nomic_data:
        datasets_to_run = list(NOMIC_DATASETS)
    elif len(datasets_to_run) == 1 and datasets_to_run[0].lower() == "all":
        datasets_to_run = DATASETS
    logger.info(f"Datasets selected for evaluation: {datasets_to_run}")

    # Initialize model
    model = UnifiedBEIRModel(
        model_name=args.model_name,
        model_type=args.model_type,
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

    metric_dir = f"{args.metric_dir}/cross_metrics/hipporag-fact/"
    os.makedirs(metric_dir, exist_ok=True)

    pred_dir = f"{args.metric_dir}/cross_predictions/hipporag-fact/"
    os.makedirs(pred_dir, exist_ok=True)

    pred_suffix = "" if args.pred_suffix is None else f"_{args.pred_suffix}"
    save_suffix = pred_suffix

    for dataset in datasets_to_run:
        data_dir = f"/data/datasets/beir/{dataset}/XC/"
        data_lbl_dir = f"/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-003/predictions/beir/{dataset}/"

        logger.info(f"\n{'='*20} Evaluating {dataset} {'='*20}")
        try:
            # 1. Load dataset
            if args.nomic_data:
                nomic_data_dir = f"/data/datasets/nomic/{NOMIC_DATASETS[dataset]}/"
                data_file = f"{nomic_data_dir}/raw_data/train.raw.csv"

                save_suffix = f"{save_suffix}_nomic"
            else:
                data_file = "{data_dir}/raw_data/test.raw.csv"

            if not os.path.exists(data_file): continue
            data_ids, data_txt = load_raw_file(data_file)

            queries = [{"id":i, "text":t} for i,t in zip(data_ids, data_txt)]

            if dataset in WIKIPEDIA_DATASETS:
                lbl_file = f"/data/datasets/beir/hotpotqa/XC/raw_data/hipporag-fact.raw.csv"
                save_suffix = f"{save_suffix}_hotpotqa"
            else:
                lbl_file = f"{data_dir}/raw_data/hipporag-fact{pred_suffix}.raw.csv"

            if not os.path.exists(lbl_file): 
                print(f"Invalid label file: {lbl_file}")
                continue

            lbl_ids, lbl_txt = load_raw_file(lbl_file)
            corpus = [{"id":i, "text":t} for i,t in zip(lbl_ids, lbl_txt)]

            data_lbl_file = f"{data_lbl_dir}/test_hipporag-fact{save_suffix}.npz"
            data_lbl = None
            if os.path.exists(data_lbl_file):
                data_lbl = retain_topk(sp.load_npz(data_lbl_file), k=5)
                data_lbl.data[:] = 1.0

            qrels = list()
            if data_lbl is not None:
                for i,row in enumerate(data_lbl):
                    for j,sc in zip(row.indices, row.data):
                        qrels.append({"query_id":data_ids[i], "corpus_id":lbl_ids[j], "score": sc})

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
            if data_lbl is not None:
                if not isinstance(qrels, dict):
                    qrels_dict = {}
                    for item in qrels:
                        qid = str(item.get("query-id", item.get("query_id")))
                        did = str(item.get("corpus-id", item.get("corpus_id")))
                        score = int(item.get("score", 1))
                        qrels_dict.setdefault(qid, {})[did] = score
                    qrels = qrels_dict

            logger.info(f"Loaded {len(queries)} queries and {len(corpus)} corpus documents.")

            # Set current dataset in the model for corpus embeddings caching.
            # Key the cache on the corpus source: Wikipedia datasets share hotpotqa's
            # fact file, everything else uses its own hipporag-fact{pred_suffix} file.
            if dataset in WIKIPEDIA_DATASETS:
                corpus_cache_tag = "hotpotqa-hipporag-fact"
            else:
                corpus_cache_tag = f"{dataset.replace('/', '-')}-hipporag-fact{pred_suffix}"
            model.set_dataset(corpus_cache_tag, corpus, args.embeddings_dir)

            # Get batch size for this dataset
            dataset_batch_size = batch_size_map.get(dataset, default_batch_size)

            # 3. Initialize BEIR dense retrieval exact search wrapper
            if args.use_anns:
                model_wrapper = HNSWSQFaissSearchCompact(model, batch_size=dataset_batch_size, 
                                                         show_progress_bar=True, corpus_chunk_size=100_000, 
                                                         use_gpu=False)
            else:
                model_wrapper = DenseRetrievalExactSearch(model, batch_size=dataset_batch_size, 
                                                          show_progress_bar=True, corpus_chunk_size=100_000)

            retriever = EvaluateRetrieval(model_wrapper, score_function="cos_sim")

            # 4. Perform retrieval
            logger.info("Starting dense retrieval...")
            results = retriever.retrieve(corpus, queries)

            dataset_prefix = dataset.replace("/", "-")
                
            lbl_ids2idx = {str(i):idx for idx,i in enumerate(lbl_ids)}
            data, indices, indptr = [], [], [0]
            for i in data_ids:
                for l, sc in results[str(i)].items():
                    data.append(sc)
                    indices.append(lbl_ids2idx[l])
                indptr.append(len(data))
            pred_lbl = sp.csr_matrix((data, indices, indptr), dtype=np.float32, shape=(len(data_ids), len(lbl_ids)))
            save_file = ( 
                f"{pred_dir}/train_predictions_{dataset_prefix}{save_suffix}.npz" 
                if args.nomic_data else 
                f"{pred_dir}/test_predictions_{dataset_prefix}{save_suffix}.npz"
            )
            sp.save_npz(save_file, pred_lbl)

            # 5. Evaluate and compute metrics
            logger.info("Computing metrics...")
            if data_lbl is not None:
                ndcg, _map, recall, precision = retriever.evaluate(
                    qrels, results, retriever.k_values, 
                )
                mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

                metrics = {}
                for name, vals in [("NDCG", ndcg), ("Recall", recall), ("Precision", precision), ("MAP", _map), ("MRR", mrr)]:
                    for k, v in vals.items(): metrics.update({f"{name}@{k.split('@')[1]}": v})

                with open(f"{metric_dir}/{dataset_prefix}{save_suffix}.json", "w") as file:
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
