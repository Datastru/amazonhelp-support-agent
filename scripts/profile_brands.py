"""
Ranks candidate brands on thread depth, volume, and apparent resolution rate,
so the brand pick is a measured decision instead of a vibe.

Resolution is a cheap proxy (last message is the brand's and reads like a
closing line) - not ground truth. Good enough to rank candidates by, noted as
a proxy in the report.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.threads import load_data, index_corpus, thread_touches_brand, is_resolved

CANDIDATES = [
    "AmazonHelp", "AppleSupport", "Uber_Support", "SpotifyCares",
    "Delta", "AmericanAir", "TMobileHelp", "comcastcares",
    "British_Airways", "Ask_Spectrum", "XboxSupport", "AskPlayStation",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=500000, help="rows to scan (perf cap)")
    ap.add_argument("--out", default="outputs/brand_profile.csv")
    args = ap.parse_args()

    df = load_data()
    if len(df) > args.sample:
        df = df.tail(args.sample).reset_index(drop=True)

    chains, fields = index_corpus(df)
    present_brands = set(df["author_id"])

    rows = []
    for brand in CANDIDATES:
        if brand not in present_brands:
            continue
        brand_chains = [c for c in chains if thread_touches_brand(c, fields, brand)]
        if not brand_chains:
            continue
        depths = [len(c) for c in brand_chains]
        resolved = sum(is_resolved(c, fields, brand) for c in brand_chains)
        rows.append({
            "brand": brand,
            "n_threads": len(brand_chains),
            "avg_turn_depth": round(sum(depths) / len(depths), 2),
            "pct_depth_4plus": round(100 * sum(d >= 4 for d in depths) / len(brand_chains), 1),
            "resolved_pct": round(100 * resolved / len(brand_chains), 1),
        })

    out = pd.DataFrame(rows).sort_values(["avg_turn_depth", "resolved_pct"], ascending=False)
    print(out.to_string(index=False))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
