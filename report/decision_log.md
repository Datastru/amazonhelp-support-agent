# Decision log

Non-obvious calls made while building this, and why. Roughly chronological.

1. **Dataset source: HF parquet mirror instead of the Kaggle API.** No
   `kaggle.json` in this environment. Found `SunidhiSriram/twcs` on the HF
   Hub with the identical schema (`tweet_id, author_id, inbound, created_at,
   text, response_tweet_id, in_response_to_tweet_id`) and row count (2.81M)
   as `thoughtvector/customer-support-on-twitter` - same data, no login
   needed. Cited in the README.

2. **Brand picked by measured profiling, not intuition.** Scored the
   assignment's suggested candidates (AmazonHelp, AppleSupport, Uber_Support,
   SpotifyCares, etc.) on thread depth, volume, and a crude resolution-rate
   proxy (`scripts/profile_brands.py`). AmazonHelp won on depth *and* volume
   decisively (avg 3.4 turns/thread, 38% reach 4+ turns, 16k+ threads in a
   500k-row sample) - see `outputs/brand_profile.csv`.

3. **Restricted to English-language tweets.** First clustering pass (on
   unfiltered data) split almost entirely along *language*, not intent -
   AmazonHelp is multilingual and short-tweet embeddings pick up language
   before topic. Filtered with `langdetect` on each thread's first customer
   turn. This is a scope cut, not a data artifact - logged under "what I
   chose not to build" in the report.

4. **Local LLM (Ollama) instead of a cloud API.** No `ANTHROPIC_API_KEY` /
   `OPENAI_API_KEY` in this environment; Ollama was already running with
   `qwen2.5:7b-instruct`, `qwen2.5:14b-instruct` pulled. Used those plus
   `nomic-embed-text` for embeddings. This is itself a "misleading headline
   number" candidate - see report section 6.

5. **Switched to Ollama's batched `/api/embed`, not `/api/embeddings`.**
   Single-item embedding calls cost ~2s each (per-request overhead, plus
   swapping the loaded model between chat and embed). Batched calls cost
   ~70ms/item - a ~28x speedup that's the only reason clustering and the
   retrieval index fit in the time budget at all.

6. **Merged 4 of 11 raw KMeans clusters into one `delivery_problem`
   intent.** Clusters for "missing package," "tracking says
   delivered/attempted, customer disputes it," "hasn't shipped yet," and
   "guaranteed date broken" were lexically close enough that a downstream
   classifier couldn't reliably separate them - and more importantly, the
   right first response is identical for all four (apologize, check status,
   offer to investigate/refund). Kept `prime_membership_value` distinct even
   though it's also delivery-related, because those customers explicitly
   frame it as a membership-value/churn question - a different escalation
   signal than a one-off late package. This merge is a human call layered on
   the clustering, not something the algorithm decided - see
   `pipeline/intents.py`.

7. **Cluster assignment doubles as the baseline classifier's training
   label.** No separate hand-labeling pass to get TF-IDF+LogReg training
   data - the KMeans cluster IS the label, and it's free (no LLM calls, no
   extra annotation round).

8. **Golden-set sampling stratifies on the baseline classifier's predicted
   intent *and* its confidence** (`scripts/build_golden_set.py`), not a flat
   random sample - guarantees every intent (including the smallest) and both
   an "easy" and "ambiguous" tier are represented. A random sample would have
   been dominated by `delivery_problem`.

9. **The near-duplicate dedup pass (cosine > 0.92) removed zero examples**
   from the golden-set sample. Worth saying plainly rather than pretending
   credit for it - the per-intent quota sampling already produced enough
   diversity on its own at this sample size.

10. **Retrieval corpus and clustering sample both capped at 1200 items**
    (out of 4572 resolved threads / 28k English threads available).
    Embedding throughput (~14 items/sec even batched) makes this the
    practical ceiling for a 15-minute reproduce budget; diminishing returns
    past this size anyway for a k=3 retrieval lookup.

11. **Escalation cost matrix is explicit, not accuracy-driven**: missing a
    safety-relevant message costs 10, missing an ordinary escalation-worthy
    one costs 3, an unnecessary escalation costs 1
    (`eval/metrics.py:COST_*`). A single accuracy or F1 number would treat
    all three the same, which is exactly backwards for this problem.

12. **Escalation policy is first-match-wins over a priority list**, not a
    weighted score - safety beats everything else regardless of what else is
    true about the message. Every decision carries one named reason string,
    not a confidence float, per the brief's own ask for something
    explainable.

13. **Split "build once" artifacts from the timed "reproduce" path.**
    Reconstructing threads for the full brand + langdetect filtering over
    ~38k threads takes ~4 minutes on its own - too much of a 15-minute
    budget to spend before any evaluation happens. The retrieval index,
    baseline classifier, and golden set are committed as artifacts; the
    README's timed reproduce path only re-runs classification/retrieval/
    drafting/judging over the golden set, not the raw-data pipeline.

14. **No fine-tuned neural classifier.** No labeled-training budget or GPU
    time justified it for a take-home. Compared TF-IDF+LogReg against a
    well-prompted LLM classifier instead, per the brief's own suggested
    alternative ("fine-tune ... or use a well-prompted LLM ... try both,
    compare").

15. **Judge defaults to the same model family as the drafter**
    (`qwen2.5:7b-instruct`) - a real, acknowledged limitation (correlated
    errors, possible leniency toward its own family's phrasing), not
    something papered over. Addressed by hand-scoring a subset and reporting
    the human/judge agreement (Cohen's kappa) rather than trusting the judge
    blindly - see `golden_set/human_judge_agreement.csv`.

16. **Escalation keyword regexes were left narrow on purpose.** Hand-labeling
    the golden set surfaced real misses - "$15k" isn't matched by the
    `\$\d+` amount pattern, "illegal" alone doesn't match the legal/PR regex
    (which looks for phrases like "legal action"), "50 times in the past
    year" doesn't match the "in a row" repeated-contact pattern. Left as-is
    rather than hand-tuned to catch these specific examples - expanding
    regexes to fit the eval set would be overfitting to the golden set
    instead of building a genuinely more robust policy. These misses show up
    honestly as false negatives in the escalation metrics.
