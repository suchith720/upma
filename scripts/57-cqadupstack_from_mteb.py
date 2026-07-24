#!/usr/bin/env python
"""
Download all 12 CQADupStack sub-datasets via the MTEB library and materialise them in the
same "XC" core format as /data/datasets/beir/cqadupstack/<subset>/XC/, i.e.:

    <subset>/XC/raw_data/test.raw.csv     # test queries  -> columns: identifier,text
    <subset>/XC/raw_data/label.raw.csv    # corpus (docs) -> columns: identifier,text  (text only, no title)
    <subset>/XC/tst_X_Y.npz               # CSR qrels matrix, shape (n_queries, n_labels), rows=test order, cols=label order
    <subset>/XC/configs/data.json         # config pointing at the three files above

NOTE: Only these BEIR-derived core files are reproducible from MTEB. The manually-created
reference folder additionally contains downstream/LLM-generated artifacts (hipporag-*,
document_concept_and_summary, document_intent_substring, document_substring,
generated-queries, examples, ...) which are NOT part of the BEIR dataset and are therefore
NOT produced here.

After generation, the script compares the generated core files against the reference folder
and prints a per-subset + overall diff report.
"""

import os
import json
import argparse
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

import mteb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# CQADupStack subset -> MTEB task name.
SUBSET_TO_TASK = {
    "android": "CQADupstackAndroidRetrieval",
    "english": "CQADupstackEnglishRetrieval",
    "gaming": "CQADupstackGamingRetrieval",
    "gis": "CQADupstackGisRetrieval",
    "mathematica": "CQADupstackMathematicaRetrieval",
    "physics": "CQADupstackPhysicsRetrieval",
    "programmers": "CQADupstackProgrammersRetrieval",
    "stats": "CQADupstackStatsRetrieval",
    "tex": "CQADupstackTexRetrieval",
    "unix": "CQADupstackUnixRetrieval",
    "webmasters": "CQADupstackWebmastersRetrieval",
    "wordpress": "CQADupstackWordpressRetrieval",
}


def load_mteb_subset(task_name: str, split: str = "test", subset: str = "default"):
    """Return (corpus_hf, queries_hf, relevant_docs) for an MTEB retrieval task."""
    task = mteb.get_tasks(tasks=[task_name])[0]
    task.load_data()
    node = task.dataset[subset][split]
    return node["corpus"], node["queries"], node["relevant_docs"]


# Reference cleaning: every C0 control char (0x00-0x1F) and DEL (0x7F) is replaced by a
# space (length-preserving). The reference label/test files contain no control characters.
_CTRL_TO_SPACE = {i: " " for i in range(0x20)}
_CTRL_TO_SPACE[0x7F] = " "


def _clean_text(t):
    return t.translate(_CTRL_TO_SPACE) if isinstance(t, str) else t


def write_raw_csv(path: str, ids, texts):
    """Write an XC raw file with columns 'identifier,text' (pandas round-trippable).

    Control characters (0x00-0x1F, 0x7F) are replaced with a space to match the reference
    dataset (which contains none) and because the csv writer cannot emit e.g. NUL bytes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = [_clean_text(t) for t in texts]
    df = pd.DataFrame({"identifier": ids, "text": clean})
    df.to_csv(path, index=False)


def build_qrels_matrix(query_ids, label_ids, relevant_docs) -> sp.csr_matrix:
    """Build a CSR matrix (n_queries, n_labels) from {qid: {docid: score}}."""
    qid2row = {str(q): i for i, q in enumerate(query_ids)}
    lid2col = {str(l): j for j, l in enumerate(label_ids)}
    rows, cols, data = [], [], []
    n_missing = 0
    for qid, docs in relevant_docs.items():
        r = qid2row.get(str(qid))
        if r is None:
            n_missing += 1
            continue
        for docid, score in docs.items():
            c = lid2col.get(str(docid))
            if c is None:
                n_missing += 1
                continue
            rows.append(r)
            cols.append(c)
            data.append(float(score))
    if n_missing:
        logger.warning(f"  {n_missing} qrel entries referenced ids not present in queries/corpus.")
    mat = sp.csr_matrix(
        (np.array(data, dtype=np.float32), (rows, cols)),
        shape=(len(query_ids), len(label_ids)),
    )
    return mat


def generate_subset(subset: str, out_root: str):
    task_name = SUBSET_TO_TASK[subset]
    logger.info(f"[{subset}] loading MTEB task {task_name} ...")
    corpus, queries, relevant_docs = load_mteb_subset(task_name)

    label_ids = [str(x) for x in corpus["id"]]
    label_txt = [p + " " + q for p,q in zip(corpus["title"], corpus["text"])]
    query_ids = [str(x) for x in queries["id"]]
    query_txt = list(queries["text"])

    xc_dir = os.path.join(out_root, subset, "XC")
    raw_dir = os.path.join(xc_dir, "raw_data")
    cfg_dir = os.path.join(xc_dir, "configs")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(cfg_dir, exist_ok=True)

    test_csv = os.path.join(raw_dir, "test.raw.csv")
    label_csv = os.path.join(raw_dir, "label.raw.csv")
    npz_path = os.path.join(xc_dir, "tst_X_Y.npz")
    cfg_path = os.path.join(cfg_dir, "data.json")

    write_raw_csv(test_csv, query_ids, query_txt)
    write_raw_csv(label_csv, label_ids, label_txt)

    mat = build_qrels_matrix(query_ids, label_ids, relevant_docs)
    sp.save_npz(npz_path, mat)

    config = {
        "data": {
            "path": {
                "test": {
                    "data_lbl": npz_path,
                    "data_info": test_csv,
                    "lbl_info": label_csv,
                }
            }
        }
    }
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=4)

    logger.info(
        f"[{subset}] wrote {len(query_ids)} queries, {len(label_ids)} labels, "
        f"qrels nnz={mat.nnz} -> {xc_dir}"
    )
    return xc_dir


def _read_raw(path):
    df = pd.read_csv(path).fillna("")
    return [str(x) for x in df["identifier"].tolist()], [str(x) for x in df["text"].tolist()]


def _qrels_triples(npz_path, test_ids, label_ids):
    """Return a set of (qid, docid, score) triples from an XC qrels matrix."""
    m = sp.load_npz(npz_path).tocoo()
    triples = set()
    for r, c, v in zip(m.row, m.col, m.data):
        triples.add((test_ids[r], label_ids[c], float(v)))
    return triples


def compare_subset(subset: str, gen_root: str, ref_root: str):
    """Compare generated vs reference XC core files. Returns a dict of findings."""
    gen = os.path.join(gen_root, subset, "XC")
    ref = os.path.join(ref_root, subset, "XC")
    findings = {"subset": subset}

    gt_ids, gt_txt = _read_raw(os.path.join(gen, "raw_data", "test.raw.csv"))
    rt_ids, rt_txt = _read_raw(os.path.join(ref, "raw_data", "test.raw.csv"))
    gl_ids, gl_txt = _read_raw(os.path.join(gen, "raw_data", "label.raw.csv"))
    rl_ids, rl_txt = _read_raw(os.path.join(ref, "raw_data", "label.raw.csv"))

    def cmp_raw(g_ids, g_txt, r_ids, r_txt):
        d = {}
        d["n_gen"], d["n_ref"] = len(g_ids), len(r_ids)
        d["same_order_ids"] = g_ids == r_ids
        d["same_id_set"] = set(g_ids) == set(r_ids)
        d["only_in_gen"] = len(set(g_ids) - set(r_ids))
        d["only_in_ref"] = len(set(r_ids) - set(g_ids))
        # per-id text equality on the shared ids
        gmap = dict(zip(g_ids, g_txt))
        rmap = dict(zip(r_ids, r_txt))
        shared = set(gmap) & set(rmap)
        text_mismatch = sum(1 for i in shared if gmap[i] != rmap[i])
        d["text_mismatch_on_shared"] = text_mismatch
        d["n_shared"] = len(shared)
        return d

    findings["queries"] = cmp_raw(gt_ids, gt_txt, rt_ids, rt_txt)
    findings["labels"] = cmp_raw(gl_ids, gl_txt, rl_ids, rl_txt)

    # qrels comparison (map through each file's own id ordering)
    gen_tr = _qrels_triples(os.path.join(gen, "tst_X_Y.npz"), gt_ids, gl_ids)
    ref_tr = _qrels_triples(os.path.join(ref, "tst_X_Y.npz"), rt_ids, rl_ids)
    findings["qrels"] = {
        "nnz_gen": len(gen_tr),
        "nnz_ref": len(ref_tr),
        "identical": gen_tr == ref_tr,
        "only_in_gen": len(gen_tr - ref_tr),
        "only_in_ref": len(ref_tr - gen_tr),
        # score-agnostic (pairs only), in case scores differ
        "pairs_identical": {(q, d) for q, d, _ in gen_tr} == {(q, d) for q, d, _ in ref_tr},
    }
    return findings


def print_report(all_findings):
    print("\n" + "=" * 30 + " COMPARISON REPORT " + "=" * 30)
    overall_ok = True
    for f in all_findings:
        sub = f["subset"]
        q, l, r = f["queries"], f["labels"], f["qrels"]
        q_ok = q["same_id_set"] and q["text_mismatch_on_shared"] == 0
        l_ok = l["same_id_set"] and l["text_mismatch_on_shared"] == 0
        r_ok = r["identical"]
        sub_ok = q_ok and l_ok and r_ok
        overall_ok = overall_ok and sub_ok
        print(f"\n### {sub}  [{'MATCH' if sub_ok else 'DIFF'}]")
        print(f"  queries: gen={q['n_gen']} ref={q['n_ref']} | same_id_set={q['same_id_set']} "
              f"same_order={q['same_order_ids']} text_mismatch={q['text_mismatch_on_shared']} "
              f"only_gen={q['only_in_gen']} only_ref={q['only_in_ref']}")
        print(f"  labels : gen={l['n_gen']} ref={l['n_ref']} | same_id_set={l['same_id_set']} "
              f"same_order={l['same_order_ids']} text_mismatch={l['text_mismatch_on_shared']} "
              f"only_gen={l['only_in_gen']} only_ref={l['only_in_ref']}")
        print(f"  qrels  : nnz gen={r['nnz_gen']} ref={r['nnz_ref']} | identical={r['identical']} "
              f"pairs_identical={r['pairs_identical']} only_gen={r['only_in_gen']} only_ref={r['only_in_ref']}")
    print("\n" + "=" * 79)
    print(f"OVERALL: {'ALL SUBSETS MATCH (core files identical)' if overall_ok else 'DIFFERENCES FOUND (see above)'}")
    print("=" * 79)


def main():
    parser = argparse.ArgumentParser(description="Reproduce CQADupStack XC core files from MTEB and compare.")
    parser.add_argument("--output_dir", default="/home/sasokan/suchith/datasets",
                        help="Root under which <subset>/XC/... is written (default: /home/sasokan/suchith/datasets).")
    parser.add_argument("--reference_dir", default="/data/datasets/beir/cqadupstack",
                        help="Reference root containing <subset>/XC/... to compare against.")
    parser.add_argument("--subsets", nargs="+", default=["all"],
                        help="Subsets to process (e.g. android gis). Use 'all' for every CQADupStack subset.")
    parser.add_argument("--compare_only", action="store_true",
                        help="Skip generation; only compare already-generated files against the reference.")
    args = parser.parse_args()

    subsets = list(SUBSET_TO_TASK) if args.subsets == ["all"] else args.subsets
    out_root = os.path.join(args.output_dir, "cqadupstack")

    all_findings = []
    for subset in subsets:
        if subset not in SUBSET_TO_TASK:
            logger.warning(f"Unknown subset '{subset}', skipping.")
            continue
        try:
            if not args.compare_only:
                generate_subset(subset, out_root)
            all_findings.append(compare_subset(subset, out_root, args.reference_dir))
        except Exception as e:
            logger.error(f"[{subset}] failed: {e}", exc_info=True)

    if all_findings:
        print_report(all_findings)


if __name__ == "__main__":
    main()
