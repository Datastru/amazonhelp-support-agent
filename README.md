# AmazonHelp support agent - Hiver take-home

Classify -> retrieve precedent -> draft grounded reply -> decide auto-handle
vs escalate, for `@AmazonHelp` on Twitter (from the Customer Support on
Twitter corpus). See `report/REPORT.md` for the write-up and
`report/decision_log.md` for the "why" behind the non-obvious calls.

## Why AmazonHelp

Picked by measured comparison, not a guess - `scripts/profile_brands.py`
scores every candidate brand on thread depth, volume, and a resolution-rate
proxy. AmazonHelp won on depth *and* volume (avg 3.4 turns/thread, 38% of
threads reach 4+ turns, largest thread count of any candidate). Numbers in
`outputs/brand_profile.csv`, discussion in the report.

## Architecture

```
tweet -> classify (TF-IDF+LogReg baseline, and a few-shot prompted LLM)
      -> retrieve top-k similar resolved threads (embedding cosine search)
      -> draft reply grounded in retrieved precedent (LLM)
      -> rule-based escalation policy (named reason, not a confidence float)
      -> output + reason
```

## Prerequisites (one-time setup, not part of the 15-minute clock)

- Python 3.12+, `pip install -r requirements.txt`
- [Ollama](https://ollama.com) running locally, with:
  ```
  ollama pull qwen2.5:7b-instruct
  ollama pull qwen2.5:14b-instruct
  ollama pull nomic-embed-text
  ```
  (~10GB total download.) The pipeline talks to Ollama's REST API on
  `localhost:11434` - no cloud API key needed or used anywhere in this repo.

## Quick reproduce (~12 minutes)

The retrieval index, baseline classifier, and hand-labeled golden set are
already committed under `artifacts/` and `golden_set/` - you don't need to
re-download or re-process the raw 2.8M-row corpus to reproduce the headline
numbers, only to re-run the live classify/retrieve/draft/judge calls over the
golden set:

```
python eval/run_eval.py --n 35
```

Prints a summary (intent accuracy, escalation precision/recall + cost,
reply-quality judge scores, all three ways: the system vs. a trivial baseline
vs. a simple baseline) and writes `outputs/eval_report.json` +
`outputs/confusion_matrix.png`. `--n 35` is a time-boxed subsample (seeded,
`--seed 42`); see the note on this in report section 6.

The numbers actually cited in `report/REPORT.md` are from the full golden set
(208 examples, ~70 minutes, already committed as
`outputs/eval_report_full.json`). Reproduce them yourself with:

```
python eval/run_eval.py --out outputs/eval_report_full.json
```

## Full rebuild from raw data (optional, ~10 extra minutes)

Only needed if you want to regenerate the committed artifacts from scratch
(different brand, different sample sizes, etc.):

```
python scripts/download_data.py                        # ~1 min, ~290MB
python scripts/profile_brands.py                        # ~15s -> outputs/brand_profile.csv
python scripts/build_threads.py --brand AmazonHelp       # ~4 min -> data/processed/
python scripts/cluster_intents.py                        # ~2.5 min -> outputs/intent_clusters.md
#   (read the cluster dump, name clusters by hand in pipeline/intents.py - already done)
python scripts/train_baseline_classifier.py               # ~5s -> artifacts/
python scripts/build_retrieval_index.py                   # ~1 min -> artifacts/
python scripts/build_golden_set.py                        # ~1 min -> golden_set/unlabeled.csv
#   (hand-label true_intent / ideal_reply_direction / escalate / escalate_reason - already done,
#    see golden_set/golden_set.csv and scripts/_apply_golden_labels.py for how)
```

## Repo layout

```
scripts/            one-off / data-pipeline CLIs (download, profile, build, train)
pipeline/            the actual system: classify, retrieval, draft, escalate, agent
eval/                 judge + metrics + the eval harness
golden_set/           200+ hand-labeled examples, sampling notes, judge-agreement check
artifacts/            committed: retrieval index, baseline classifier, intent training labels
outputs/               generated: eval_report.json, brand_profile.csv, confusion_matrix.png
report/                 REPORT.md (the write-up) and decision_log.md
```

## What this doesn't do

Not multilingual (English-only, see decision log #3), not fine-tuned (TF-IDF
baseline vs. a prompted LLM, no neural classifier training), not connected to
real order/account data (public tweets only - see report section 6). All
deliberate scope cuts, not oversights - see report section 1 and the decision
log.
