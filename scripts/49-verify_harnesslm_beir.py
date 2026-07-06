#!/usr/bin/env python
"""
Standalone script to verify/compare student and teacher text embedding models.
Takes in a TSV query file, passes queries through student and teacher encoders,
computes L2 loss, MSE, and Cosine Similarity, and prints/saves the results.
"""

import os
import argparse
import logging
import json
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class DataParallelEncoder(torch.nn.Module):
    def __init__(self, st_model):
        super().__init__()
        self.st_model = st_model

    def forward(self, **features):
        return self.st_model(features)["sentence_embedding"]


class EmbeddingModelWrapper:
    """
    Wrapper for SentenceTransformer models to handle custom prefixes, device setting, and PyTorch DataParallel.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "auto",
        query_prefix: str = None,
        device: str = None,
        use_data_parallel: bool = False,
    ):
        logger.info(f"Loading model: {model_name} on device: {device}")
        self.model_name = model_name.lower()
        self.device = device
        self.use_data_parallel = use_data_parallel

        # Prepare kwargs
        st_kwargs = {"trust_remote_code": True, "device": device}
        model_kwargs = {}
        tokenizer_kwargs = {}
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

        logger.info(f"Loading SentenceTransformer with kwargs: {st_kwargs}")
        self.model = SentenceTransformer(model_name, **st_kwargs)
        self.model.max_seq_length = 512

        # Data Parallelism
        if self.use_data_parallel and torch.cuda.device_count() > 1:
            logger.info(f"Initializing torch.nn.DataParallel across {torch.cuda.device_count()} GPUs...")
            self.parallel_model = torch.nn.DataParallel(DataParallelEncoder(self.model))
            self.parallel_model.to(device)
        else:
            self.use_data_parallel = False
            self.parallel_model = None

        # Resolve model type
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

        # Set default query prefix
        if self.model_type == "nomic":
            self.query_prefix = "search_query: " if query_prefix is None else query_prefix
        elif self.model_type == "kalm":
            self.query_prefix = (
                "Instruct: Given a query, retrieve documents that answer the query \n Query: "
                if query_prefix is None
                else query_prefix
            )
        elif self.model_type == "qwen":
            self.query_prefix = (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
                if query_prefix is None
                else query_prefix
            )
        else:
            self.query_prefix = "" if query_prefix is None else query_prefix

        logger.info(f"Using Query Prefix: {repr(self.query_prefix)}")

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

    def encode_queries(self, queries: List[str], batch_size: int = 16) -> np.ndarray:
        logger.info(f"Encoding {len(queries)} queries with batch size {batch_size}...")
        prefixed_queries = [f"{self.query_prefix}{q}" for q in queries]
        return self._encode_texts(self.model, self.parallel_model, prefixed_queries, batch_size=batch_size)


def load_queries(file_path: str, query_column: str = None, no_header: bool = False) -> List[str]:
    logger.info(f"Loading queries from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Query file {file_path} does not exist.")
    
    if no_header:
        df = pd.read_table(file_path, header=None)
        queries = df[0].astype(str).tolist()
    else:
        df = pd.read_table(file_path)
        if query_column:
            if query_column in df.columns:
                queries = df[query_column].astype(str).tolist()
            else:
                raise KeyError(f"Column '{query_column}' not found in query file. Columns present: {list(df.columns)}")
        else:
            # Look for common column names case-insensitively
            possible_cols = ["query", "text", "query_text", "raw_data", "raw_text", "raw_query"]
            found_col = None
            for col in df.columns:
                if col.lower() in possible_cols:
                    found_col = col
                    break
            
            if found_col:
                logger.info(f"Automatically identified query column: '{found_col}'")
                queries = df[found_col].astype(str).tolist()
            else:
                # Fallback to the first column
                first_col = df.columns[0]
                logger.warning(f"No common query column found. Using the first column: '{first_col}'")
                queries = df[first_col].astype(str).tolist()
                
    logger.info(f"Successfully loaded {len(queries)} queries.")
    return queries


def main():
    parser = argparse.ArgumentParser(description="Compare student and teacher embeddings on queries and compute L2 loss.")
    parser.add_argument(
        "--query_file",
        type=str,
        required=True,
        help="Path to the TSV file containing queries.",
    )
    parser.add_argument(
        "--query_column",
        type=str,
        default=None,
        help="Column name in the TSV file representing queries. If None, auto-detects.",
    )
    parser.add_argument(
        "--no_header",
        action="store_true",
        help="Specify if the TSV file has no header row.",
    )
    parser.add_argument(
        "--student_model_name",
        type=str,
        default="nomic-ai/nomic-embed-text-v1",
        help="HuggingFace model repository name or path for the student tower.",
    )
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default="nomic-ai/nomic-embed-text-v1",
        help="HuggingFace model repository name or path for the teacher tower.",
    )
    parser.add_argument(
        "--student_model_type",
        type=str,
        default="auto",
        choices=["auto", "nomic", "kalm", "qwen", "custom"],
        help="Student model type to apply default prompts (nomic, kalm, qwen, custom). 'auto' resolves based on model name.",
    )
    parser.add_argument(
        "--teacher_model_type",
        type=str,
        default="auto",
        choices=["auto", "nomic", "kalm", "qwen", "custom"],
        help="Teacher model type to apply default prompts (nomic, kalm, qwen, custom). 'auto' resolves based on model name.",
    )
    parser.add_argument(
        "--student_query_prefix",
        type=str,
        default=None,
        help="Override/provide student query prefix instruction.",
    )
    parser.add_argument(
        "--teacher_query_prefix",
        type=str,
        default=None,
        help="Override/provide teacher query prefix instruction.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for model encoding.",
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
        "--output_file",
        type=str,
        default=None,
        help="Path to save the comparison results as JSON (optional).",
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

    # Load queries
    queries = load_queries(args.query_file, query_column=args.query_column, no_header=args.no_header)

    # Initialize models
    logger.info("Initializing student model...")
    student_model = EmbeddingModelWrapper(
        model_name=args.student_model_name,
        model_type=args.student_model_type,
        query_prefix=args.student_query_prefix,
        device=device,
        use_data_parallel=args.use_data_parallel,
    )

    logger.info("Initializing teacher model...")
    teacher_model = EmbeddingModelWrapper(
        model_name=args.teacher_model_name,
        model_type=args.teacher_model_type,
        query_prefix=args.teacher_query_prefix,
        device=device,
        use_data_parallel=args.use_data_parallel,
    )

    # Encode queries
    rnd_idx = np.random.permutation(len(queries))[:1000]

    logger.info("Encoding queries with student model...")
    student_embeddings = student_model.encode_queries([queries[i] for i in rnd_idx], batch_size=args.batch_size)
    # student_embeddings = student_embeddings[:, :128]
    # student_embeddings /= np.maximum(
    #     np.linalg.norm(student_embeddings, axis=-1, keepdims=True),
    #     1e-12
    # )

    logger.info("Encoding queries with teacher model...")
    teacher_embeddings = teacher_model.encode_queries([queries[i] for i in rnd_idx], batch_size=args.batch_size)
    # teacher_embeddings = teacher_embeddings[:, :128]
    # teacher_embeddings /= np.maximum(
    #     np.linalg.norm(teacher_embeddings, axis=-1, keepdims=True),
    #     1e-12
    # )

    student_dim = student_embeddings.shape[1]
    teacher_dim = teacher_embeddings.shape[1]
    logger.info(f"Student dimension: {student_dim}, Teacher dimension: {teacher_dim}")

    # Dimension mismatch check and slice
    if student_dim != teacher_dim:
        logger.warning(
            f"Dimension mismatch! Student dimension ({student_dim}) != Teacher dimension ({teacher_dim}). "
            "Truncating the larger embeddings to match the smaller dimensions for L2/similarity computation."
        )
        min_dim = min(student_dim, teacher_dim)
        student_embeddings = student_embeddings[:, :min_dim]
        teacher_embeddings = teacher_embeddings[:, :min_dim]

    # Compute comparison metrics
    logger.info("Computing metrics...")

    diff = np.sum((student_embeddings - teacher_embeddings) ** 2, axis=-1)
    mse = float(diff.mean())

    # Print results
    logger.info(f"Log Mean Squared Error (MSE): {np.log10(mse):.8f}")
    logger.info(f"{'='*54}\n")

if __name__ == "__main__":
    main()
