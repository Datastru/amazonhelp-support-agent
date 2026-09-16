"""Embeds the resolved-thread corpus and writes artifacts/retrieval_index.npy
+ artifacts/retrieval_meta.jsonl - the precedent set the drafter retrieves
against. Committed to the repo so eval doesn't need to redo the embedding
pass every run."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.retrieval import build_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/resolved_pairs.jsonl")
    ap.add_argument("--cap", type=int, default=1200)
    args = ap.parse_args()

    n = build_index(args.pairs, cap=args.cap)
    print(f"indexed {n} resolved threads -> artifacts/retrieval_index.npy")


if __name__ == "__main__":
    main()
