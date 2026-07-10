#!/usr/bin/env python
"""
Convert a contrastors `BiEncoder` checkpoint into a self-contained
SentenceTransformer directory that `43-evaluate_beir.py` (and any
`SentenceTransformer(path, trust_remote_code=True)` caller) can load.

A trained contrastors checkpoint is just:
    config.json      # a BiEncoderConfig dump: architectures=["BiEncoder"], no model_type / auto_map
    model.safetensors  # weights, all under the "trunk." prefix

SentenceTransformer cannot load that (no model_type, no auto_map, no modules.json /
pooling config). The trunk, however, is architecturally identical to the public
`nomic-ai/nomic-embed-text-v1` HF model (verified: same 112 weight tensors, same
shapes). So we graft the fine-tuned trunk weights onto the public repo's
self-contained scaffolding (custom modeling code + config with auto_map + tokenizer
+ modules.json + 1_Pooling), producing a drop-in SentenceTransformer model.

Usage (run in the `nomic` conda env):
    python 55-convert_biencoder_to_st.py \
        --ckpt /data/suchith/outputs/benchmarks/06-nomic_embed_text_v1_reproduce-001/step_9000/model \
        --out  /data/suchith/outputs/benchmarks/06-nomic_embed_text_v1_reproduce-001/step_9000/st_model
"""

import argparse
import logging
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TRUNK_PREFIX = "trunk."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Path to the contrastors BiEncoder checkpoint dir (has config.json + model.safetensors)")
    p.add_argument("--out", required=True, help="Output directory for the SentenceTransformer model")
    p.add_argument("--base-repo", default="nomic-ai/nomic-embed-text-v1", help="Public HF repo providing the ST scaffolding (default: nomic-ai/nomic-embed-text-v1)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite --out if it already exists")
    p.add_argument("--skip-sanity", action="store_true", help="Skip the SentenceTransformer load/encode sanity check")
    return p.parse_args()


def load_trunk_state_dict(ckpt_dir):
    st_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"No model.safetensors found in {ckpt_dir}")
    sd = load_file(st_path)
    trunk_sd = {k[len(TRUNK_PREFIX):]: v for k, v in sd.items() if k.startswith(TRUNK_PREFIX)}
    if not trunk_sd:
        raise ValueError(
            f"No '{TRUNK_PREFIX}' keys in {st_path}. Is this really a contrastors BiEncoder checkpoint? "
            f"Found top-level prefixes: {sorted({k.split('.')[0] for k in sd})}"
        )
    non_trunk = [k for k in sd if not k.startswith(TRUNK_PREFIX)]
    if non_trunk:
        # proj / hamming are nn.Identity (no params) for this model; warn if anything real shows up
        logger.warning(f"Ignoring {len(non_trunk)} non-trunk tensor(s): {non_trunk}")
    logger.info(f"Loaded {len(trunk_sd)} trunk tensors from checkpoint")
    return trunk_sd


def main():
    args = parse_args()

    if os.path.exists(args.out):
        if not args.overwrite:
            raise FileExistsError(f"{args.out} already exists. Pass --overwrite to replace it.")
        logger.info(f"Removing existing {args.out}")
        shutil.rmtree(args.out)

    # 1. Fine-tuned trunk weights.
    trunk_sd = load_trunk_state_dict(args.ckpt)

    # 2. Verify the fine-tuned trunk is exactly compatible with the base HF model BEFORE
    #    we write anything, so a mismatch fails loudly instead of producing a broken model.
    logger.info(f"Loading base model {args.base_repo} to validate key compatibility...")
    hf = AutoModel.from_pretrained(args.base_repo, trust_remote_code=True)
    missing, unexpected = hf.load_state_dict(trunk_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "State-dict mismatch between checkpoint trunk and base model:\n"
            f"  missing (base expects, ckpt lacks): {list(missing)}\n"
            f"  unexpected (ckpt has, base lacks): {list(unexpected)}\n"
            "Refusing to write a partial model."
        )
    logger.info("Trunk weights load into the base architecture with strict key match. Good.")

    # 3. Materialize the base repo's full ST scaffolding (modeling code, config w/ auto_map,
    #    tokenizer, modules.json, 1_Pooling, sentence_bert_config.json) into --out.
    logger.info(f"Downloading ST scaffolding from {args.base_repo}...")
    snap = snapshot_download(args.base_repo)
    # copytree with symlinks resolved so --out is standalone (HF cache stores blobs via symlink)
    shutil.copytree(snap, args.out, symlinks=False, ignore=shutil.ignore_patterns(".cache", "*.h5", "*.msgpack", "onnx", "*.onnx"))
    logger.info(f"Copied scaffolding to {args.out}")

    # 4. Overwrite the weights with our fine-tuned trunk. Drop any stale weight shards
    #    from the base repo so ST doesn't load the wrong file.
    for stale in ("pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"):
        fp = os.path.join(args.out, stale)
        if os.path.exists(fp):
            os.remove(fp)
            logger.info(f"Removed stale {stale}")

    trunk_sd = {k: v.contiguous() for k, v in trunk_sd.items()}
    save_file(trunk_sd, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    logger.info(f"Wrote fine-tuned weights to {os.path.join(args.out, 'model.safetensors')}")

    # 5. Sanity check: load via SentenceTransformer and confirm it embeds + that a matching
    #    query/document pair scores higher than a mismatched one (prefixes applied manually).
    if not args.skip_sanity:
        logger.info("Sanity-checking with SentenceTransformer...")
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim

        model = SentenceTransformer(args.out, trust_remote_code=True)
        model.max_seq_length = 512
        q = "search_query: what is the capital of France?"
        d_pos = "search_document: Paris is the capital and most populous city of France."
        d_neg = "search_document: The mitochondrion is the powerhouse of the cell."
        emb = model.encode([q, d_pos, d_neg], convert_to_tensor=True, normalize_embeddings=True)
        pos = cos_sim(emb[0], emb[1]).item()
        neg = cos_sim(emb[0], emb[2]).item()
        logger.info(f"cos(query, positive)={pos:.4f}  cos(query, negative)={neg:.4f}  dim={emb.shape[1]}")
        if pos <= neg:
            logger.warning("Positive did not outscore negative — inspect the model before trusting eval numbers.")
        else:
            logger.info("Sanity check passed (positive > negative).")

    logger.info(f"\nDone. SentenceTransformer model ready at:\n    {args.out}\n"
                f"Run BEIR eval with:\n"
                f"    python 43-evaluate_beir.py --model_name {args.out} --model_type nomic --datasets all --use_data_parallel ...")


if __name__ == "__main__":
    main()
