"""
Intent discovery: embed a sample of customer root messages, KMeans, dump the
closest-to-centroid examples per cluster so a human can read and name them.
Picking k by silhouette score over a small range rather than guessing.

Output feeds two things by hand afterwards:
  1. pipeline/intents.py - the final named taxonomy (edited by hand after
     reading outputs/intent_clusters.md)
  2. artifacts/training_labels.csv - cluster id -> message, used as silver
     labels to train the TF-IDF baseline classifier (no LLM calls needed to
     generate training data - the cluster assignment IS the label)
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.llm_client import embed

MENTION_RE = re.compile(r"^@\S+\s*")


def clean(text):
    return MENTION_RE.sub("", text).strip()


def load_customer_messages(path, n, seed=42):
    msgs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            thread = json.loads(line)
            first = next((t for t in thread["turns"] if t["inbound"]), None)
            if first and len(first["text"]) > 15:
                msgs.append(clean(first["text"]))
    rng = np.random.default_rng(seed)
    if len(msgs) > n:
        idx = rng.choice(len(msgs), size=n, replace=False)
        msgs = [msgs[i] for i in idx]
    return msgs


def pick_k(X, k_range):
    best_k, best_score = k_range[0], -1
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X)
        score = silhouette_score(X, km.labels_, sample_size=min(2000, len(X)))
        print(f"  k={k}  silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="data/processed/threads.jsonl")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--k-min", type=int, default=8)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--out-md", default="outputs/intent_clusters.md")
    ap.add_argument("--out-csv", default="artifacts/training_labels.csv")
    args = ap.parse_args()

    msgs = load_customer_messages(args.threads, args.n)
    print(f"embedding {len(msgs)} customer messages")
    X = np.array(embed(msgs))

    print("picking k:")
    k = pick_k(X, range(args.k_min, args.k_max + 1))
    print(f"using k={k}")

    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    labels = km.labels_

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with open(args.out_md, "w", encoding="utf-8") as f:
        for c in range(k):
            idx = np.where(labels == c)[0]
            center = km.cluster_centers_[c]
            dists = np.linalg.norm(X[idx] - center, axis=1)
            closest = idx[np.argsort(dists)[:8]]
            f.write(f"## cluster {c}  (n={len(idx)})\n\n")
            for i in closest:
                f.write(f"- {msgs[i]}\n")
            f.write("\n")
    print(f"wrote {args.out_md} - read it, name each cluster in pipeline/intents.py")

    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "cluster_id"])
        for msg, c in zip(msgs, labels):
            w.writerow([msg, int(c)])
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
