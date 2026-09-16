"""Cosine-similarity retrieval over the resolved-thread corpus. No FAISS -
a few thousand vectors is small enough that brute-force numpy is plenty fast
and it's one less dependency."""
import json
import os

import numpy as np

from pipeline.llm_client import embed

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
VECS_PATH = os.path.join(INDEX_DIR, "retrieval_index.npy")
META_PATH = os.path.join(INDEX_DIR, "retrieval_meta.jsonl")


def build_index(pairs_path, out_dir=INDEX_DIR, cap=1200, seed=42):
    """pairs_path: resolved_pairs.jsonl (issue_text, resolution_reply).
    Caps corpus size - embedding is the slow step, and a few thousand
    precedents is already more diversity than a k-NN lookup needs."""
    records = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    rng = np.random.default_rng(seed)
    if len(records) > cap:
        idx = rng.choice(len(records), size=cap, replace=False)
        records = [records[i] for i in idx]

    texts = [r["issue_text"] for r in records]
    vecs = np.array(embed(texts), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "retrieval_index.npy"), vecs)
    with open(os.path.join(out_dir, "retrieval_meta.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return len(records)


class Index:
    def __init__(self, vecs_path=VECS_PATH, meta_path=META_PATH):
        self.vecs = np.load(vecs_path)
        self.meta = [json.loads(l) for l in open(meta_path, encoding="utf-8")]

    def retrieve(self, query_text, k=3):
        q = np.array(embed(query_text)[0], dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-8
        sims = self.vecs @ q
        top_idx = np.argsort(-sims)[:k]
        return [
            {**self.meta[i], "similarity": float(sims[i])}
            for i in top_idx
        ]
