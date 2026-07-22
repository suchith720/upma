#!/usr/bin/env python
"""
Standalone script to evaluate **asymmetric two-tower** text embedding models on BEIR
retrieval tasks using the **MTEB** library.

This is the asymmetric-tower analogue of ``56-evaluate_beir_mteb.py``: instead of a single
model encoding both queries and documents, it uses **two separate SentenceTransformer
towers** -- a query tower and a document tower (as in ``48-evaluate_harnesslm_beir.py``).
MTEB drives retrieval + metric computation; the towers are exposed through a thin
``mteb.models.AbsEncoder`` wrapper that routes each ``encode(...)`` call to the correct
tower based on ``prompt_type`` (query vs. document).

Key asymmetric behaviours (ported from 48-evaluate_harnesslm_beir.py):
  * Query tower encodes queries (with the query prefix); document tower encodes the corpus
    (with the document prefix).
  * If the document tower's embedding dim is larger than the query tower's, document
    embeddings are truncated to the query dim so the two spaces are comparable.
  * ``--project`` truncates the final embeddings (both sides) to 128 dims.

Notes:
  * No local datasets are used -- everything is downloaded via MTEB from HuggingFace.
  * MSMARCO uses MTEB's default eval split, which is ``dev`` (public MSMARCO has no test qrels).
  * Quora (symmetric duplicate-question retrieval) encodes its corpus with the *query* tower
    (see ``DOC_AS_QUERY_TASKS`` / ``current_dataset == "quora"``), mirroring 43/48/56.
  * Document (corpus) embeddings are cached to disk, keyed on the doc tower + MTEB task +
    split, so re-runs skip re-encoding the corpus.
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

# BEIR dataset name -> MTEB task name (same mapping as 56-evaluate_beir_mteb.py).
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

# Tasks whose corpus should be encoded with the *query* tower (symmetric retrieval),
# matching the `doc_as_query` handling in 43/48/56.
DOC_AS_QUERY_TASKS = {"QuoraRetrieval"}

# k cutoffs and metric families reported in the BEIR-style output (matches 56-evaluate_beir_mteb.py).
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


def _resolve_model_type(model_name: str, model_type: str) -> str:
    """Auto-detect the model family (nomic/kalm/qwen/custom) from the model name."""
    if model_type != "auto":
        return model_type.lower()
    name = model_name.lower()
    if "nomic" in name:
        return "nomic"
    if "kalm" in name:
        return "kalm"
    if "qwen" in name:
        return "qwen"
    return "custom"


def _build_st_kwargs(model_name: str, device: str) -> dict:
    """SentenceTransformer load kwargs (qwen padding side + 8B dtype), matching 48/56."""
    st_kwargs = {"trust_remote_code": True, "device": device}
    model_kwargs = {}
    tokenizer_kwargs = {}
    name = model_name.lower()
    if "qwen" in name:
        tokenizer_kwargs["padding_side"] = "left"
        if "8b" in name:
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
    return st_kwargs


class DataParallelEncoder(torch.nn.Module):
    def __init__(self, st_model):
        super().__init__()
        self.st_model = st_model

    def forward(self, **features):
        return self.st_model(features)["sentence_embedding"]


class AsymmetricMTEBEncoder(AbsEncoder):
    """
    MTEB (v2) encoder wrapper around a **pair** of SentenceTransformer towers.

    Routes each ``encode(...)`` call to the query tower or the document tower based on
    ``prompt_type``, applying the appropriate prefix, then harmonises the embedding
    dimensions (truncating the doc tower to the query dim; optional 128-d projection) so
    MTEB can compute cosine similarity between the two spaces.
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
        batch_size: int = 256,
        use_data_parallel: bool = False,
        embeddings_dir: str = None,
        project: bool = False,
    ):
        self.query_model_name = query_model_name.lower()
        self.doc_model_name = doc_model_name.lower()
        self.device = device
        self.batch_size = batch_size
        self.embeddings_dir = embeddings_dir
        self.use_data_parallel = use_data_parallel
        self.project = project

        # ---- load the two towers ----
        query_st_kwargs = _build_st_kwargs(query_model_name, device)
        doc_st_kwargs = _build_st_kwargs(doc_model_name, device)
        logger.info(f"Loading query tower: {query_model_name} with kwargs: {query_st_kwargs}")
        self.query_model = SentenceTransformer(query_model_name, **query_st_kwargs)
        self.query_model.max_seq_length = 512
        logger.info(f"Loading doc tower: {doc_model_name} with kwargs: {doc_st_kwargs}")
        self.doc_model = SentenceTransformer(doc_model_name, **doc_st_kwargs)
        self.doc_model.max_seq_length = 512

        # ---- optional DataParallel ----
        if self.use_data_parallel and torch.cuda.device_count() > 1:
            logger.info(f"Initializing torch.nn.DataParallel across {torch.cuda.device_count()} GPUs...")
            self.parallel_query_model = torch.nn.DataParallel(DataParallelEncoder(self.query_model)).to(device)
            self.parallel_doc_model = torch.nn.DataParallel(DataParallelEncoder(self.doc_model)).to(device)
        else:
            self.use_data_parallel = False
            self.parallel_query_model = None
            self.parallel_doc_model = None

        # ---- resolve model types + prefixes ----
        self.query_model_type = _resolve_model_type(self.query_model_name, query_model_type)
        self.doc_model_type = _resolve_model_type(self.doc_model_name, doc_model_type)
        logger.info(f"Query tower type: {self.query_model_type}; Doc tower type: {self.doc_model_type}")

        # Query prefix follows the QUERY tower's family.
        if self.query_model_type == "nomic":
            self.query_prefix = "search_query: " if query_prefix is None else query_prefix
        elif self.query_model_type == "kalm":
            self.query_prefix = (
                "Instruct: Given a query, retrieve documents that answer the query \n Query: "
                if query_prefix is None else query_prefix
            )
        elif self.query_model_type == "qwen":
            self.query_prefix = (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
                if query_prefix is None else query_prefix
            )
        else:
            self.query_prefix = "" if query_prefix is None else query_prefix

        # Doc prefix follows the DOC tower's family.
        if self.doc_model_type == "nomic":
            self.doc_prefix = "search_document: " if doc_prefix is None else doc_prefix
        else:
            self.doc_prefix = "" if doc_prefix is None else doc_prefix

        logger.info(f"Using Query Prefix: {repr(self.query_prefix)}")
        logger.info(f"Using Document Prefix: {repr(self.doc_prefix)}")

        # ---- embedding dims (for asymmetric truncation) ----
        self.query_dim = self.query_model.get_sentence_embedding_dimension()
        self.doc_dim = self.doc_model.get_sentence_embedding_dimension()
        logger.info(f"Query dim: {self.query_dim}; Doc dim: {self.doc_dim}; project128: {self.project}")

        # Per-task flags (set by main() before each task).
        self.doc_as_query = False
        self.current_dataset = None

        # MTEB plumbing. We apply prefixes ourselves (not via model_prompts) and force
        # cosine similarity (see `similarity`) to match BEIR's cos_sim.
        self.model_prompts = None
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(
                name=f"{query_model_name}__{doc_model_name}",
                revision="main",
            )
        )

    def similarity(self, embeddings1, embeddings2):
        """Force cosine similarity (matches BEIR's score_function='cos_sim')."""
        return cos_sim(embeddings1, embeddings2)

    def _cache_path(self, task_name, hf_subset, hf_split, prompt_type):
        if self.embeddings_dir is None:
            return None
        sanitized_model = self.doc_model_name.strip("/").replace("/", "_").replace("\\", "_")
        subset = (hf_subset or "default").replace("/", "-")
        split = (hf_split or "test").replace("/", "-")
        ptype = prompt_type.value if isinstance(prompt_type, PromptType) else str(prompt_type)
        proj = "_project" if self.project else ""
        fname = f"{task_name.replace('/', '-')}_{subset}_{split}_{ptype}_{sanitized_model}{proj}.npy"
        return os.path.join(self.embeddings_dir, fname)

    def _encode_texts(self, model, parallel_model, texts: List[str], batch_size: int) -> np.ndarray:
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
        # Symmetric-retrieval tasks (e.g. Quora) encode the corpus with the QUERY tower.
        doc_as_query = self.doc_as_query or (self.current_dataset == "quora")
        use_query_tower = is_query or doc_as_query

        batch_size = int(kwargs.get("batch_size", self.batch_size) or self.batch_size)

        # Cache only corpus embeddings that come from the DOC tower (the expensive, reusable
        # side). Queries -- and Quora's query-tower corpus -- are not cached.
        cache_path = None
        if not is_query and not doc_as_query:
            cache_path = self._cache_path(task_metadata.name, hf_subset, hf_split, prompt_type)
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

        if use_query_tower:
            prefix, model, parallel = self.query_prefix, self.query_model, self.parallel_query_model
            tower = "query"
        else:
            prefix, model, parallel = self.doc_prefix, self.doc_model, self.parallel_doc_model
            tower = "doc"

        logger.info(
            f"Encoding {len(texts)} {'queries' if is_query else 'documents'} for "
            f"task={task_metadata.name} split={hf_split} via {tower}-tower "
            f"(prefix={prefix!r}, batch={batch_size})..."
        )
        prefixed = [f"{prefix}{t}" for t in texts]
        embeddings = self._encode_texts(model, parallel, prefixed, batch_size=batch_size)

        # Asymmetric dim harmonisation: truncate doc-tower embeddings to the query dim so the
        # two towers live in a comparable space (ported from 48-evaluate_harnesslm_beir.py).
        if not use_query_tower and embeddings.shape[1] > self.query_dim:
            logger.info(f"Truncating doc embeddings {embeddings.shape[1]} -> {self.query_dim} (query dim)")
            embeddings = embeddings[:, :self.query_dim]
        if self.project:
            embeddings = embeddings[:, :128]

        if cache_path is not None:
            os.makedirs(self.embeddings_dir, exist_ok=True)
            logger.info(f"Saving embeddings to {cache_path}...")
            np.save(cache_path, embeddings)

        return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate asymmetric two-tower embedding models on BEIR tasks via MTEB."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact"],
        help="List of BEIR datasets to evaluate (e.g. arguana scifact). Use 'all' for all standard BEIR datasets.",
    )
    parser.add_argument(
        "--query_model_name",
        type=str,
        default="nomic-ai/nomic-embed-text-v1",
        help="HuggingFace model repository name for the QUERY tower.",
    )
    parser.add_argument(
        "--doc_model_name",
        type=str,
        default="nomic-ai/nomic-embed-text-v1",
        help="HuggingFace model repository name for the DOCUMENT tower.",
    )
    parser.add_argument(
        "--query_model_type",
        type=str,
        default="auto",
        choices=["auto", "nomic", "kalm", "qwen", "custom"],
        help="Query tower model type to apply default prompts. 'auto' resolves based on model name.",
    )
    parser.add_argument(
        "--doc_model_type",
        type=str,
        default="auto",
        choices=["auto", "nomic", "kalm", "qwen", "custom"],
        help="Document tower model type to apply default prompts. 'auto' resolves based on model name.",
    )
    parser.add_argument("--query_prefix", type=str, default=None, help="Override/provide query prefix instruction.")
    parser.add_argument("--doc_prefix", type=str, default=None, help="Override/provide document prefix instruction.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where MTEB writes its raw per-task result files. If None, MTEB does not persist them.",
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
        "--project",
        action="store_true",
        help="Project the final embeddings (both towers) to 128 dimensions.",
    )
    parser.add_argument(
        "--metric_dir",
        type=str,
        default=None,
        help="Directory to save per-dataset metric JSON files and the collated beir.json. If None, nothing is written.",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=None,
        help="Directory to save/load cached document (corpus) embeddings. If None, caching is disabled.",
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

    # Initialize the asymmetric two-tower encoder
    model = AsymmetricMTEBEncoder(
        query_model_name=args.query_model_name,
        doc_model_name=args.doc_model_name,
        query_model_type=args.query_model_type,
        doc_model_type=args.doc_model_type,
        query_prefix=args.query_prefix,
        doc_prefix=args.doc_prefix,
        device=device,
        batch_size=default_batch_size,
        use_data_parallel=args.use_data_parallel,
        embeddings_dir=embeddings_dir,
        project=args.project,
    )

    metric_dir = None
    if args.metric_dir is not None:
        metric_dir = f"{args.metric_dir}/metrics"
        os.makedirs(metric_dir, exist_ok=True)
    if args.output_dir is not None:
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
            # Per-dataset query instruction for a qwen query tower (optional; needs xcai).
            # if model.query_model_type == "qwen" and args.query_prefix is None:
            #     from xcai.maggi.utils import DATASETS, get_instruction
            #     instruction_file = "/home/sasokan/suchith/xcai/xcai/models/nvembed/instructions.json"
            #     instruction = get_instruction(instruction_file, DATASETS[dataset])["query"]
            #     model.query_prefix = f"Instruct: {instruction} \n Query: "
            #     logger.info(f"Using qwen query prefix: {model.query_prefix!r}")

            # Symmetric retrieval: encode corpus with the query tower for Quora.
            model.doc_as_query = task_name in DOC_AS_QUERY_TASKS
            model.current_dataset = dataset  # 43/48-style dataset-name hook (e.g. "quora")

            model.batch_size = batch_size_map.get(dataset, default_batch_size)

            task = mteb.get_tasks(tasks=[task_name])
            evaluation = mteb.MTEB(tasks=task)
            # NB: no explicit eval_splits -> MTEB uses each task's default; MSMARCO -> 'dev'.
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

            if metric_dir is not None:
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

    if metric_dir is not None:
        collate_beir_metrics(metric_dir)


if __name__ == "__main__":
    main()
