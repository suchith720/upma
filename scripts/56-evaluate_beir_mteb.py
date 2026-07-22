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
import gc
import json
import hashlib
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

# Silence the benign "leaked semaphore" UserWarning that Python's resource_tracker prints at
# shutdown when SentenceTransformer's multi-process pool is torn down (workers are hard-
# terminated, so their queue semaphores are reclaimed only by the tracker's shutdown pass --
# no real leak survives the process). The tracker runs in a SEPARATE process, so it must be
# silenced via PYTHONWARNINGS (inherited at that child's startup), not a runtime filter.
_rt_filter = "ignore::UserWarning:multiprocessing.resource_tracker"
os.environ["PYTHONWARNINGS"] = (
    os.environ["PYTHONWARNINGS"] + "," + _rt_filter if os.environ.get("PYTHONWARNINGS") else _rt_filter
)

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


def _release_pool_queues(pool):
    """Promptly close a multi-process pool's queues so their semaphores are cleaned up now,
    avoiding a benign 'leaked semaphore' warning from the resource tracker at shutdown."""
    if isinstance(pool, dict):
        for key in ("input", "output"):
            q = pool.get(key)
            if q is not None:
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass


def _resolve_dtype(device, dtype):
    """Map a --dtype choice to a torch dtype for model loading.
    'auto' -> bf16 (if supported) / fp16 on CUDA, fp16 on MPS, fp32 (None) elsewhere."""
    dtype = (dtype or "auto").lower()
    if dtype == "fp32":
        return None  # SentenceTransformer default (float32)
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    # auto
    if device == "cuda" and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return None


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
        multi_process: bool = False,
        dtype: str = "auto",
    ):
        st_kwargs = {"trust_remote_code": True, "device": device}
        model_kwargs = {}
        tokenizer_kwargs = {}
        self.model_name = model_name.lower()
        if "qwen" in self.model_name:
            tokenizer_kwargs["padding_side"] = "right"

        # Load in reduced precision (default bf16/fp16 on CUDA) for ~2x speed + half memory.
        torch_dtype = _resolve_dtype(device, dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        self.dtype_tag = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(torch_dtype, "fp32")

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
        self.multi_process = multi_process
        self.pool = None
        self.parallel_model = None

        # Multi-GPU encoding: prefer the ST multi-process pool (one process per GPU,
        # near-linear speedup). Falls back to (inferior) DataParallel, then single-GPU.
        if self.multi_process and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.use_data_parallel = False
            target_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            logger.info(f"Starting multi-process encode pool on {target_devices}...")
            self.pool = self.model.start_multi_process_pool(target_devices=target_devices)
        else:
            if self.multi_process:
                logger.warning("--multi_process needs >=2 CUDA GPUs; using single-GPU encode.")
                self.multi_process = False
            if self.use_data_parallel and torch.cuda.device_count() > 1:
                logger.info(f"Initializing torch.nn.DataParallel across {torch.cuda.device_count()} GPUs...")
                self.parallel_model = torch.nn.DataParallel(DataParallelEncoder(self.model))
                self.parallel_model.to(device)
            else:
                self.use_data_parallel = False

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
        # BEIR dataset name for the task currently being evaluated (e.g. "quora"). Mirrors
        # 43-evaluate_beir.py's `self.current_dataset == "quora"` doc-as-query hook.
        self.current_dataset = None

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


    def _cache_path(self, task_name, hf_subset, hf_split, prompt_type, texts):
        if self.embeddings_dir is None:
            return None
        sanitized_model = self.model_name.strip("/").replace("/", "_").replace("\\", "_")
        subset = (hf_subset or "default").replace("/", "-")
        split = (hf_split or "test").replace("/", "-")
        ptype = prompt_type.value if isinstance(prompt_type, PromptType) else str(prompt_type)
        # Content hash of THIS chunk's texts. MTEB calls encode() once per corpus CHUNK, so a
        # key that ignores the chunk contents collides across chunks -- returning the wrong
        # (first-chunk) embeddings for every same-size chunk. Hashing the texts (+ dtype) makes
        # each chunk its own cache entry: correct, and still reused across identical re-runs.
        h = hashlib.md5()
        h.update(str(len(texts)).encode())
        for t in texts:
            h.update(b"\x1f")
            h.update(t.encode("utf-8", "ignore"))
        digest = h.hexdigest()[:16]
        fname = f"{task_name.replace('/', '-')}_{subset}_{split}_{ptype}_{sanitized_model}_{self.dtype_tag}_{digest}.npy"
        return os.path.join(self.embeddings_dir, fname)

    def _encode_texts(self, texts: List[str], batch_size: int) -> np.ndarray:
        if self.multi_process and self.pool is not None:
            # One process per GPU; ST shards the text list across the pool. Returns numpy.
            # (encode_multi_process is deprecated in ST v5+; encode() now accepts `pool`.)
            return self.model.encode(
                texts,
                pool=self.pool,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
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

    def close(self):
        """Tear down the multi-process pool, if any (call once at the end)."""
        if self.pool is not None:
            logger.info("Stopping multi-process encode pool...")
            try:
                self.model.stop_multi_process_pool(self.pool)
            except Exception as e:
                logger.warning(f"Error stopping multi-process pool: {e}")
            _release_pool_queues(self.pool)
            self.pool = None
            gc.collect()

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
        # Two hooks (either triggers): the per-task DOC_AS_QUERY_TASKS flag, and 43's
        # dataset-name check `self.current_dataset == "quora"`.
        doc_as_query = self.doc_as_query or (self.current_dataset == "quora")
        use_query_prefix = is_query or (not is_query and doc_as_query)
        prefix = self.query_prefix if use_query_prefix else self.doc_prefix

        # Prefer our configured batch size; MTEB otherwise injects its own default (32),
        # which would starve the GPUs. `self.batch_size` is set per-dataset in main().
        batch_size = int(self.batch_size or kwargs.get("batch_size") or 32)

        # Cache only document embeddings (queries carry per-task instructions that may vary,
        # and are cheap relative to the corpus) -- mirrors 43-evaluate_beir.py.
        cache_path = None if is_query else self._cache_path(task_metadata.name, hf_subset, hf_split, prompt_type, texts)
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
        default=None,
        help="Directory where MTEB writes its raw per-task result files. If None, MTEB does not persist them.",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="512",
        help="Batch size for encoding. Can be an integer or a dict mapping dataset names to integer sizes.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. 'cuda', 'mps', 'cpu'). Auto-detects if None.")
    parser.add_argument("--use_data_parallel", action="store_true", help="Use torch.nn.DataParallel across GPUs (slower; prefer --multi_process).")
    parser.add_argument(
        "--multi_process",
        action="store_true",
        help="Encode with a SentenceTransformer multi-process pool (one process per GPU) for near-linear multi-GPU speedup.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Model load precision. 'auto' = bf16 (if supported) / fp16 on CUDA, else fp32. Lower precision is faster.",
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

    # Enable TF32 for faster fp32 matmuls on Ampere+ GPUs (free speedup, minimal accuracy impact).
    torch.set_float32_matmul_precision("high")

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
        multi_process=args.multi_process,
        dtype=args.dtype,
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
            # Per-task query instruction for qwen (matches 43-evaluate_beir.py).
            # if model.model_type == "qwen":
            #     from xcai.maggi.utils import DATASETS, get_instruction
            #     instruction_file = "/home/sasokan/suchith/xcai/xcai/models/nvembed/instructions.json"
            #     instruction = get_instruction(instruction_file, DATASETS[dataset])["query"]
            #     model.query_prefix = f"Instruct: {instruction} \n Query: "
            #     logger.info(f"Using qwen query prefix: {model.query_prefix!r}")

            # Symmetric retrieval: encode corpus with the query prefix for Quora.
            model.doc_as_query = task_name in DOC_AS_QUERY_TASKS
            # Track the BEIR dataset name so the encoder's `current_dataset == "quora"`
            # hook (ported from 43-evaluate_beir.py) also fires.
            model.current_dataset = dataset

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

    # Tear down the multi-process encode pool (if one was started).
    model.close()


if __name__ == "__main__":
    main()
