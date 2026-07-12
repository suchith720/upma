#!/usr/bin/env python
"""
Standalone script to evaluate text embedding models (e.g. Nomic Embed, KaLM-Embedding,
Qwen3-Embedding, or other custom SentenceTransformer models) on BEIR retrieval tasks.

Unlike ``43-evaluate_beir.py`` (which uses the ``beir`` library for retrieval + metric
computation), this script uses the **MTEB** library to run retrieval and compute metrics
-- mirroring ``contrastors/src/contrastors/eval/eval_beir.py``. The embedding model is
exposed to MTEB through a thin ``mteb.models.AbsEncoder`` wrapper that reuses the same
prompt-prefix / DataParallel encoding logic as ``43-evaluate_beir.py``.

Document (corpus) embeddings are cached to disk, keyed on the MTEB task + split, so
re-runs of the same model/task skip re-encoding the corpus.
"""

import os
import ast
import json
import argparse
import logging
from typing import List, Dict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.similarity_functions import cos_sim
from mteb.types import PromptType

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# BEIR dataset name (as used in 43-evaluate_beir.py) -> MTEB task name.
BEIR_TO_MTEB = {
    "arguana": "ArguAna",
    "scidocs": "SCIDOCS",
    "scifact": "SciFact",
    "webis-touche2020": "Touche2020",
    "trec-covid": "TRECCOVID",
    "cqadupstack/android": "CQADupstackAndroidRetrieval",
    "cqadupstack/english": "CQADupstackEnglishRetrieval",
    "cqadupstack/gaming": "CQADupstackGamingRetrieval",
    "cqadupstack/gis": "CQADupstackGisRetrieval",
    "cqadupstack/mathematica": "CQADupstackMathematicaRetrieval",
    "cqadupstack/physics": "CQADupstackPhysicsRetrieval",
    "cqadupstack/programmers": "CQADupstackProgrammersRetrieval",
    "cqadupstack/stats": "CQADupstackStatsRetrieval",
    "cqadupstack/tex": "CQADupstackTexRetrieval",
    "cqadupstack/unix": "CQADupstackUnixRetrieval",
    "cqadupstack/webmasters": "CQADupstackWebmastersRetrieval",
    "cqadupstack/wordpress": "CQADupstackWordpressRetrieval",
    "fiqa": "FiQA2018",
    "quora": "QuoraRetrieval",
    "msmarco": "MSMARCO",
    "climate-fever": "ClimateFEVER",
    "dbpedia-entity": "DBPedia",
    "fever": "FEVER",
    "hotpotqa": "HotpotQA",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
}

# Standard BEIR datasets (order preserved for `--datasets all`)
BEIR_DATASETS = list(BEIR_TO_MTEB.keys())

# Tasks whose corpus should be encoded with the *query* prefix (symmetric retrieval),
# matching the `doc_as_query` handling in contrastors/eval_beir.py.
DOC_AS_QUERY_TASKS = {"QuoraRetrieval"}

# k cutoffs and metric families reported in the BEIR-style output (matches 43-evaluate_beir.py).
K_VALUES = [1, 3, 5, 10, 100, 1000]
# (output name, MTEB score-key prefix)
METRIC_FAMILIES = [
    ("NDCG", "ndcg"),
    ("Recall", "recall"),
    ("Precision", "precision"),
    ("MAP", "map"),
    ("MRR", "mrr"),
]


def mteb_scores_to_beir_format(result) -> Dict[str, float]:
    """Convert an MTEB TaskResult into the compact BEIR-style metric dict, e.g.
    {"NDCG@10": 0.38343, "Recall@10": 0.4539, ...}. Scores are averaged across
    subsets/splits by TaskResult.get_score and rounded to 5 decimals."""
    metrics = {}
    for out_name, mteb_prefix in METRIC_FAMILIES:
        for k in K_VALUES:
            key = f"{mteb_prefix}_at_{k}"
            try:
                val = result.get_score(getter=lambda s, key=key: s[key])
            except Exception:
                continue
            metrics[f"{out_name}@{k}"] = round(float(val), 5)
    return metrics


def collate_beir_metrics(metric_dir: str):
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


class UnifiedMTEBEncoder(AbsEncoder):
    """
    MTEB (v2) encoder wrapper around a SentenceTransformer model.

    Reuses the prompt-prefix and DataParallel encoding logic of ``UnifiedBEIRModel`` in
    ``43-evaluate_beir.py``, but conforms to MTEB's ``AbsEncoder`` interface: MTEB drives
    retrieval + metric computation, calling ``encode(...)`` once per (task, prompt_type)
    with a ``DataLoader`` of inputs and using the inherited ``similarity`` for search.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "auto",
        query_prefix: str = None,
        doc_prefix: str = None,
        device: str = None,
        batch_size: int = 256,
        use_data_parallel: bool = False,
        embeddings_dir: str = None,
    ):
        st_kwargs = {"trust_remote_code": True, "device": device}
        model_kwargs = {}
        tokenizer_kwargs = {}
        self.model_name = model_name.lower()
        if "qwen" in self.model_name:
            tokenizer_kwargs["padding_side"] = "left"
            if "8b" in self.model_name:
                if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    model_kwargs["torch_dtype"] = torch.bfloat16
                elif device == "cuda" and torch.cuda.is_available():
                    model_kwargs["torch_dtype"] = torch.float16
                elif device in ("mps", "cuda"):
                    model_kwargs["torch_dtype"] = torch.float16

        if model_kwargs:
            st_kwargs["model_kwargs"] = model_kwargs
        if tokenizer_kwargs:
            st_kwargs["tokenizer_kwargs"] = tokenizer_kwargs

        logger.info(f"Loading SentenceTransformer model: {model_name} on device: {device} with kwargs: {st_kwargs}")
        self.model = SentenceTransformer(model_name, **st_kwargs)
        self.model.max_seq_length = 512
        self.device = device
        self.batch_size = batch_size
        self.embeddings_dir = embeddings_dir
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
            elif "qwen" in self.model_name:
                self.model_type = "qwen"
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
        elif self.model_type == "qwen":
            self.query_prefix = (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query \n Query: "
                if query_prefix is None
                else query_prefix
            )
            self.doc_prefix = "" if doc_prefix is None else doc_prefix
        else:
            self.query_prefix = "" if query_prefix is None else query_prefix
            self.doc_prefix = "" if doc_prefix is None else doc_prefix

        logger.info(f"Using Query Prefix: {repr(self.query_prefix)}")
        logger.info(f"Using Document Prefix: {repr(self.doc_prefix)}")

        # Per-task flag: encode the corpus with the query prefix (symmetric retrieval).
        self.doc_as_query = False

        # MTEB requires `model_prompts` and `mteb_model_meta`. We apply prefixes ourselves
        # (not via model_prompts). Similarity is forced to cosine (see `similarity`) to
        # match BEIR's cos_sim, independent of the underlying model's default.
        self.model_prompts = None
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(
                name=model_name,
                revision="main",
            )
        )

    def similarity(self, embeddings1, embeddings2):
        """Force cosine similarity (matches BEIR's score_function='cos_sim')."""
        return cos_sim(embeddings1, embeddings2)


    def _cache_path(self, task_name, hf_subset, hf_split, prompt_type):
        if self.embeddings_dir is None:
            return None
        sanitized_model = self.model_name.strip("/").replace("/", "_").replace("\\", "_")
        subset = (hf_subset or "default").replace("/", "-")
        split = (hf_split or "test").replace("/", "-")
        ptype = prompt_type.value if isinstance(prompt_type, PromptType) else str(prompt_type)
        fname = f"{task_name.replace('/', '-')}_{subset}_{split}_{ptype}_{sanitized_model}.npy"
        return os.path.join(self.embeddings_dir, fname)

    def _encode_texts(self, texts: List[str], batch_size: int) -> np.ndarray:
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
                convert_to_numpy=True,
            )

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType = None,
        **kwargs,
    ) -> np.ndarray:
        # MTEB passes a DataLoader whose batches expose a "text" field (corpus documents
        # already have title+text combined by MTEB).
        texts = [text for batch in inputs for text in batch["text"]]

        is_query = prompt_type == PromptType.query
        # Symmetric-retrieval tasks (e.g. Quora) encode the corpus with the query prefix.
        use_query_prefix = is_query or (not is_query and self.doc_as_query)
        prefix = self.query_prefix if use_query_prefix else self.doc_prefix

        batch_size = int(kwargs.get("batch_size", self.batch_size) or self.batch_size)

        # Cache only document embeddings (queries carry per-task instructions that may vary,
        # and are cheap relative to the corpus) -- mirrors 43-evaluate_beir.py.
        cache_path = None if is_query else self._cache_path(task_metadata.name, hf_subset, hf_split, prompt_type)
        if cache_path is not None and os.path.exists(cache_path):
            logger.info(f"Loading cached embeddings from {cache_path}...")
            try:
                cached = np.load(cache_path)
                if len(cached) == len(texts):
                    return cached
                logger.warning(
                    f"Cached embeddings size ({len(cached)}) != inputs ({len(texts)}). Re-encoding."
                )
            except Exception as e:
                logger.error(f"Error loading cached embeddings: {e}. Re-encoding.", exc_info=True)

        logger.info(
            f"Encoding {len(texts)} {'queries' if is_query else 'documents'} for "
            f"task={task_metadata.name} split={hf_split} with batch size {batch_size} "
            f"(prefix={prefix!r})..."
        )
        prefixed = [f"{prefix}{t}" for t in texts]
        embeddings = self._encode_texts(prefixed, batch_size=batch_size)

        if cache_path is not None:
            os.makedirs(self.embeddings_dir, exist_ok=True)
            logger.info(f"Saving embeddings to {cache_path}...")
            np.save(cache_path, embeddings)

        return embeddings


def main():
    parser = argparse.ArgumentParser(description="Evaluate text embedding models on BEIR tasks via MTEB.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact"],
        help="List of BEIR datasets to evaluate (e.g. arguana scifact). Use 'all' for all standard BEIR datasets.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nomic-ai/nomic-embed-text-v1",
        help="HuggingFace model repository name (e.g. 'nomic-ai/nomic-embed-text-v1', 'Qwen/Qwen3-Embedding-0.6B').",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "nomic", "kalm", "qwen", "custom"],
        help="Model type to apply default prompts. 'auto' resolves based on model name.",
    )
    parser.add_argument("--query_prefix", type=str, default=None, help="Override/provide query prefix instruction.")
    parser.add_argument("--doc_prefix", type=str, default=None, help="Override/provide document prefix instruction.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./beir_evaluation/mteb_results",
        help="Directory where MTEB writes its raw per-task result files.",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="256",
        help="Batch size for encoding. Can be an integer or a dict mapping dataset names to integer sizes.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. 'cuda', 'mps', 'cpu'). Auto-detects if None.")
    parser.add_argument("--use_data_parallel", action="store_true", help="Use torch.nn.DataParallel across GPUs.")
    parser.add_argument(
        "--metric_dir",
        type=str,
        default="./beir_evaluation/metrics",
        help="Directory to save per-dataset metric JSON files and the collated beir.json.",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="./beir_evaluation/corpus_embeddings",
        help="Directory to save/load cached document (corpus) embeddings. Pass '' to disable caching.",
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
        batch_size_map = ast.literal_eval(args.batch_size)
        default_batch_size = batch_size_map.get("default", 256)

    # Determine datasets to evaluate
    datasets_to_run = args.datasets
    if len(datasets_to_run) == 1 and datasets_to_run[0].lower() == "all":
        datasets_to_run = BEIR_DATASETS
    logger.info(f"Datasets selected for evaluation: {datasets_to_run}")

    embeddings_dir = args.embeddings_dir if args.embeddings_dir else None

    # Initialize model / encoder
    model = UnifiedMTEBEncoder(
        model_name=args.model_name,
        model_type=args.model_type,
        query_prefix=args.query_prefix,
        doc_prefix=args.doc_prefix,
        device=device,
        batch_size=default_batch_size,
        use_data_parallel=args.use_data_parallel,
        embeddings_dir=embeddings_dir,
    )

    metric_dir = f"{args.metric_dir}/metrics"
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    ndcg_scores = {}

    for dataset in datasets_to_run:
        if dataset not in BEIR_TO_MTEB:
            logger.warning(f"Unknown BEIR dataset '{dataset}' (no MTEB task mapping). Skipping.")
            continue
        task_name = BEIR_TO_MTEB[dataset]
        dataset_prefix = dataset.replace("/", "-")

        logger.info(f"\n{'='*20} Evaluating {dataset} -> {task_name} {'='*20}")
        try:
            # Per-task query instruction for qwen (matches 43-evaluate_beir.py).
            if model.model_type == "qwen":
                from xcai.maggi.utils import DATASETS, get_instruction
                instruction_file = "/home/sasokan/suchith/xcai/xcai/models/nvembed/instructions.json"
                instruction = get_instruction(instruction_file, DATASETS[dataset])["query"]
                model.query_prefix = f"Instruct: {instruction} \n Query: "
                logger.info(f"Using qwen query prefix: {model.query_prefix!r}")

            # Symmetric retrieval: encode corpus with the query prefix for Quora.
            model.doc_as_query = task_name in DOC_AS_QUERY_TASKS

            model.batch_size = batch_size_map.get(dataset, default_batch_size)

            task = mteb.get_tasks(tasks=[task_name])
            evaluation = mteb.MTEB(tasks=task)
            results = evaluation.run(
                model,
                output_folder=args.output_dir,
                overwrite_results=True,
                verbosity=1,
            )

            # Extract metrics from the TaskResult and convert to the BEIR-style format.
            result = results[0]
            metrics = mteb_scores_to_beir_format(result)
            ndcg10 = metrics.get("NDCG@10")
            if ndcg10 is None:
                ndcg10 = float(result.get_score())  # fallback: main_score (ndcg_at_10 for retrieval)

            with open(f"{metric_dir}/{dataset_prefix}.json", "w") as file:
                json.dump({dataset: metrics}, file, indent=4)

            ndcg_scores[dataset_prefix] = ndcg10
            logger.info(f"{dataset}: NDCG@10 = {ndcg10:.5f}")

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
