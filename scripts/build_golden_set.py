"""
Stratified sample for hand-labeling. Two axes:
  - intent (via the baseline classifier's predicted cluster, so every named
    intent gets covered, not just the common ones)
  - easy vs ambiguous (via the baseline's prediction confidence - low
    confidence = genuinely hard/boundary case, which is exactly what a random
    sample under-represents)

Then a diversity pass: embed the candidates and greedily drop near-duplicates
(cosine > 0.92) so the set isn't the same 15 complaints restated 200 times.

Writes golden_set/unlabeled.csv - hand-labeling (intent, ideal reply
direction, escalate + reason) happens after, directly in that file.
"""
import argparse
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.intents import CLUSTER_TO_INTENT
from pipeline.llm_client import embed

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MENTION_RE = re.compile(r"^@\S+\s*")


def load_pool(threads_path):
    rows = []
    with open(threads_path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            first = next((x for x in t["turns"] if x["inbound"]), None)
            if first is None or len(first["text"]) < 15:
                continue
            rows.append({
                "thread_id": t["thread_id"],
                "text": MENTION_RE.sub("", first["text"]).strip(),
                "n_turns": len(t["turns"]),
                "resolved": t.get("resolved", False),
            })
    return pd.DataFrame(rows)


def predict_with_confidence(texts):
    with open(os.path.join(ARTIFACT_DIR, "baseline_classifier.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vec = pickle.load(f)
    X = vec.transform(texts)
    proba = clf.predict_proba(X)
    pred_idx = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    classes = clf.classes_
    intents = [CLUSTER_TO_INTENT.get(str(classes[i]), "unclear") for i in pred_idx]
    return intents, conf


def dedup_by_similarity(df, threshold=0.92, seed=42):
    vecs = np.array(embed(df["text"].tolist()), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    keep = []
    kept_vecs = []
    for i in order:
        v = vecs[i]
        if kept_vecs and max(np.array(kept_vecs) @ v) > threshold:
            continue
        keep.append(i)
        kept_vecs.append(v)
    return df.iloc[sorted(keep)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="data/processed/threads.jsonl")
    ap.add_argument("--target", type=int, default=220, help="pre-dedup pool size")
    ap.add_argument("--per-intent", type=int, default=24)
    ap.add_argument("--out", default="golden_set/unlabeled.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = load_pool(args.threads)
    print(f"{len(pool)} candidate messages")

    intents, conf = predict_with_confidence(pool["text"].tolist())
    pool["pred_intent"] = intents
    pool["confidence"] = conf

    rng = np.random.default_rng(args.seed)
    picked = []
    for intent, group in pool.groupby("pred_intent"):
        group = group.sort_values("confidence")
        n = min(args.per_intent, len(group))
        half = n // 2
        # bottom half by confidence = ambiguous, top half = easy
        ambiguous = group.iloc[:half]
        easy = group.iloc[-(n - half):]
        picked.append(pd.concat([ambiguous, easy]))
    sample = pd.concat(picked).drop_duplicates(subset="thread_id").reset_index(drop=True)
    print(f"{len(sample)} after per-intent easy/ambiguous stratification")

    sample = dedup_by_similarity(sample)
    print(f"{len(sample)} after near-duplicate removal")

    sample["difficulty"] = np.where(sample["confidence"] < sample["confidence"].median(), "ambiguous", "easy")
    for col in ["true_intent", "ideal_reply_direction", "escalate", "escalate_reason"]:
        sample[col] = ""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(f"wrote {args.out} - hand-label the true_intent/ideal_reply_direction/escalate/escalate_reason columns")


if __name__ == "__main__":
    main()
