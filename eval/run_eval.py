"""
Runs the system (classify+retrieve+draft+escalate) and both baselines over
the golden set, scores everything, writes outputs/eval_report.json +
outputs/confusion_matrix.png.

--n subsamples the golden set (seeded) so a quick run fits a time budget -
the full run (no --n) is what report/REPORT.md cites as the headline numbers.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.agent import run_agent, run_trivial_baseline, run_simple_baseline
from pipeline.retrieval import Index
from pipeline.intents import INTENTS
from eval.judge import judge_reply
from eval.metrics import intent_metrics, escalation_metrics

INTENT_LABELS = sorted(INTENTS.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="golden_set/golden_set.csv")
    ap.add_argument("--n", type=int, default=None, help="subsample size, default = full set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/eval_report.json")
    args = ap.parse_args()

    df = pd.read_csv(args.golden)
    df["escalate"] = df["escalate"].astype(bool)
    if args.n and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"evaluating on {len(df)} golden examples")

    index = Index()

    rows = []
    t0 = time.time()
    for i, r in df.iterrows():
        text = r["text"]
        sys_out = run_agent(text, index)
        simple_out = run_simple_baseline(text, index)
        trivial_out = run_trivial_baseline(text)

        judge_sys = judge_reply(text, sys_out["reply"], sys_out["retrieved"]) if sys_out["reply"] else None
        judge_simple = judge_reply(text, simple_out["reply"], simple_out["retrieved"]) if simple_out["reply"] else None

        rows.append({
            "thread_id": r["thread_id"],
            "text": text,
            "true_intent": r["true_intent"],
            "true_escalate": bool(r["escalate"]),
            "true_escalate_reason": r.get("escalate_reason", ""),
            "sys_baseline_intent": sys_out["baseline_intent"],
            "sys_llm_intent": sys_out["llm_intent"],
            "sys_escalate": sys_out["escalate"],
            "sys_escalate_trigger": sys_out["escalate_trigger"],
            "sys_reply": sys_out["reply"],
            "sys_judge": judge_sys,
            "simple_intent": simple_out["intent"],
            "simple_escalate": simple_out["escalate"],
            "simple_reply": simple_out["reply"],
            "simple_judge": judge_simple,
            "trivial_intent": trivial_out["intent"],
            "trivial_escalate": trivial_out["escalate"],
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(df)}  ({elapsed:.0f}s elapsed, {elapsed / (i + 1):.1f}s/ex)")

    out_df = pd.DataFrame(rows)

    report = {"n": len(out_df), "seconds": round(time.time() - t0, 1)}

    # --- intent: LLM classifier vs TF-IDF baseline vs trivial majority-class ---
    report["intent_llm"] = intent_metrics(out_df["true_intent"], out_df["sys_llm_intent"], INTENT_LABELS)
    report["intent_baseline_tfidf"] = intent_metrics(out_df["true_intent"], out_df["sys_baseline_intent"], INTENT_LABELS)
    report["intent_trivial"] = intent_metrics(out_df["true_intent"], out_df["trivial_intent"], INTENT_LABELS)

    # --- escalation: policy vs keyword baseline vs escalate-nothing ---
    report["escalation_policy"] = escalation_metrics(out_df["true_escalate"], out_df["sys_escalate"], out_df["true_escalate_reason"])
    report["escalation_keyword_baseline"] = escalation_metrics(out_df["true_escalate"], out_df["simple_escalate"], out_df["true_escalate_reason"])
    report["escalation_never"] = escalation_metrics(out_df["true_escalate"], out_df["trivial_escalate"], out_df["true_escalate_reason"])

    # --- reply quality: judge scores, system (grounded) vs template baseline ---
    def avg_judge(col):
        scores = [s for s in out_df[col] if isinstance(s, dict)]
        dims = ["grounded", "tone", "factual", "actionable"]
        return {d: round(np.mean([s[d] for s in scores if s[d] is not None]), 2) for d in dims}

    report["reply_quality_system"] = avg_judge("sys_judge")
    report["reply_quality_template_baseline"] = avg_judge("simple_judge")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    out_df.to_json(args.out.replace(".json", "_rows.jsonl"), orient="records", lines=True)
    print(f"wrote {args.out}")

    # confusion matrix for the LLM classifier (the "real" system, not the baseline)
    cm = np.array(report["intent_llm"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(INTENT_LABELS)), INTENT_LABELS, rotation=90)
    ax.set_yticks(range(len(INTENT_LABELS)), INTENT_LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(INTENT_LABELS)):
        for j in range(len(INTENT_LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    cm_path = args.out.replace(".json", "_confusion_matrix.png")
    fig.savefig(cm_path, dpi=120)
    print(f"wrote {cm_path}")

    print("\n--- summary ---")
    print(f"intent accuracy  llm={report['intent_llm']['report']['accuracy']:.3f}"
          f"  tfidf={report['intent_baseline_tfidf']['report']['accuracy']:.3f}"
          f"  trivial={report['intent_trivial']['report']['accuracy']:.3f}")
    print(f"escalation f1    policy={report['escalation_policy']['f1']}"
          f"  keyword={report['escalation_keyword_baseline']['f1']}"
          f"  never={report['escalation_never']['f1']}")
    print(f"escalation cost  policy={report['escalation_policy']['avg_cost_per_example']}"
          f"  keyword={report['escalation_keyword_baseline']['avg_cost_per_example']}"
          f"  never={report['escalation_never']['avg_cost_per_example']}")
    print(f"reply quality    system={report['reply_quality_system']}"
          f"  template={report['reply_quality_template_baseline']}")


if __name__ == "__main__":
    main()
