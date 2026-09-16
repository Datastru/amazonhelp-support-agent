"""
Reconstructs full conversation threads for one brand and writes two files:

  data/processed/threads.jsonl        every thread that touches the brand
  data/processed/resolved_pairs.jsonl issue -> final agent reply, for
                                       threads that look resolved (this is
                                       the retrieval corpus for drafting)

--cap limits how many raw rows we scan, so this stays well inside the 15-min
reproduce budget instead of walking the full 2.8M-row corpus every run.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.threads import (
    load_data, index_corpus, thread_touches_brand, is_resolved, is_english_thread,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def chain_to_thread(chain, fields):
    return {
        "thread_id": chain[0],
        "turns": [
            {
                "tweet_id": t,
                "author_id": fields["author"][t],
                "inbound": bool(fields["inbound"][t]),
                "text": fields["text"][t],
                "created_at": fields["created_at"][t],
            }
            for t in chain
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True)
    ap.add_argument("--cap", type=int, default=1500000, help="rows to scan")
    args = ap.parse_args()

    df = load_data()
    if len(df) > args.cap:
        df = df.tail(args.cap).reset_index(drop=True)

    chains, fields = index_corpus(df)
    brand_chains = [c for c in chains if thread_touches_brand(c, fields, args.brand)]
    print(f"{len(brand_chains)} threads touch {args.brand} (scanned {len(df)} rows)")

    brand_chains = [c for c in brand_chains if is_english_thread(c, fields)]
    print(f"{len(brand_chains)} of those are English (langdetect on first customer turn)")

    out_dir = os.path.abspath(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    n_resolved = 0
    threads_path = os.path.join(out_dir, "threads.jsonl")
    pairs_path = os.path.join(out_dir, "resolved_pairs.jsonl")
    with open(threads_path, "w", encoding="utf-8") as tf, open(pairs_path, "w", encoding="utf-8") as pf:
        for chain in brand_chains:
            thread = chain_to_thread(chain, fields)
            tf.write(json.dumps(thread) + "\n")

            resolved = is_resolved(chain, fields, args.brand)
            thread["resolved"] = resolved
            if resolved and len(chain) >= 2:
                first_customer_turn = next(t for t in thread["turns"] if t["inbound"])
                pf.write(json.dumps({
                    "thread_id": thread["thread_id"],
                    "issue_text": first_customer_turn["text"],
                    "resolution_reply": thread["turns"][-1]["text"],
                    "n_turns": len(chain),
                }) + "\n")
                n_resolved += 1

    print(f"{n_resolved} resolved threads -> retrieval corpus ({pairs_path})")
    print(f"wrote {threads_path}")


if __name__ == "__main__":
    main()
