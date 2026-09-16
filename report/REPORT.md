# Report: AmazonHelp support agent

Numbers below are from the full 208-example golden set
(`outputs/eval_report_full.json`), reproducible per the README.

## 1. Problem framing

**What "good" means for this brand.** AmazonHelp's job on Twitter isn't to
resolve orders in the reply itself - almost nothing in a 280-character public
tweet actually can be (no order-number lookup, no account access). Looking at
the resolved threads in the corpus, the brand's own playbook is: acknowledge,
apologize where warranted, and move the customer to a DM to actually handle
the account-specific part. So "good" here is narrower than "solves the
problem": a good automated reply (1) picks the right intent so it's routed
sensibly, (2) matches that playbook - acknowledge + correct next step,
grounded in how this brand actually talks, not invented policy - and (3)
correctly recognizes when it *shouldn't* attempt a reply at all.

**What we chose not to build:**
- Multilingual support. AmazonHelp is multilingual; we scoped to English only
  after the first (unfiltered) clustering pass showed embeddings splitting
  almost entirely along language rather than intent (decision log #3).
- Full order/account resolution. No access to real order data, so replies
  can only match the brand's own "acknowledge + move to DM" pattern, not
  actually resolve anything - see section 6.
- A fine-tuned classifier. TF-IDF+LogReg baseline vs. a well-prompted LLM
  classifier, compared head to head, per the brief's own suggested
  alternative to fine-tuning.
- A learned escalation model. Rule-based and explainable on purpose - every
  decision has one named reason, not a probability.

## 2. System

```
tweet -> classify (TF-IDF+LogReg, and a few-shot prompted qwen2.5:7b-instruct)
      -> retrieve top-3 similar resolved threads (nomic-embed-text, cosine)
      -> draft reply grounded in retrieved precedent (qwen2.5:7b-instruct)
      -> rule-based escalation policy (named reason)
      -> output + reason
```

8 intents, discovered by embedding 1200 customer messages and KMeans
(k=11 by silhouette), then named by reading the cluster dump and merging 4
lexically-close delivery clusters into one (`pipeline/intents.py`, decision
log #6):

| intent | description |
|---|---|
| delivery_problem | late, missing, mis-tracked, or falsely-marked-delivered package |
| prime_membership_value | late delivery framed as a Prime-value/churn complaint |
| refund_or_return_request | refund/return/replacement wanted and not yet processed |
| poor_support_experience | complaint about the support interaction itself |
| payment_or_giftcard_issue | gift card, payment method, EMI, unexpected charge |
| packaging_or_damage_complaint | damaged item or excessive packaging |
| device_or_app_tech_support | Alexa/Fire TV/app/account bug |
| general_inquiry_or_other | questions, praise, off-topic, catch-all |

Escalation triggers, checked in priority order, first match wins
(`pipeline/escalate.py`): safety/self-harm mention > legal/PR-sensitive
language > refund/compensation ask >= $100 > repeated-unresolved-contact
phrasing > no close precedent in the retrieval index (similarity < 0.45) >
baseline/LLM classifier disagreement. Anything else auto-handles.

## 3. Results vs. baselines

Three systems compared on the full golden set:
- **trivial**: majority-class intent (`delivery_problem`), one canned reply,
  never escalates
- **simple**: TF-IDF intent, template reply (no retrieval grounding),
  keyword-only escalation (safety/legal regex only)
- **system**: the full pipeline described above

| | intent accuracy | intent macro-F1 | escalation P / R / F1 | escalation avg cost/ex | reply quality (avg of 4 dims) |
|---|---|---|---|---|---|
| trivial | 0.130 | 0.029 | 0 / 0 / 0 (never escalates) | 0.389 | n/a (canned reply, not judged) |
| simple (TF-IDF + keyword) | 0.466 | 0.475 | 0 / 0 / 0 | 0.389 | grounded 1.91, tone 3.60, factual 2.56, actionable 2.18 |
| system | **0.688** | **0.680** | 0.150 / 0.593 / 0.239 | 0.596 | grounded 2.43, tone 4.00, factual 3.03, actionable 3.06 |

(full per-class precision/recall/F1 and the confusion matrix are in
`outputs/eval_report_full.json` / `outputs/eval_report_full_confusion_matrix.png`)

**Intent**: the LLM classifier clearly beats the TF-IDF baseline (0.688 vs
0.466 accuracy) and both crush the trivial majority-class baseline (0.130) -
expected, since 8 balanced-in-the-golden-set classes make "always guess
`delivery_problem`" a bad strategy even though delivery_problem is the
single largest class in the real traffic distribution.

**Reply quality**: the retrieval-grounded system beats the ungrounded
template baseline on every dimension, most on `actionable` (3.06 vs 2.18)
and `factual` (3.03 vs 2.56) - makes sense, the template baseline is one
fixed sentence that's never wrong but never specific either, so it scores
low on anything requiring it to say something concrete.

**Escalation is the one where the headline number actively misleads** - see
section 6, this needed real digging and is worth reading before trusting the
0.596 cost figure at face value.

## 4. Golden set

208 hand-labeled examples (`golden_set/golden_set.csv`), stratified across
all 8 intents (26 each) and easy/ambiguous difficulty (104/104) by the
baseline classifier's own confidence - see `golden_set/sampling_notes.md` for
the full sampling and labeling methodology.

**Diversity, not just count.** Reading through all 208 by hand while
labeling: diversity is uneven across intents.
`device_or_app_tech_support` and `payment_or_giftcard_issue` are almost
entirely distinct issues (a different Alexa quirk or gift-card question each
time). `packaging_or_damage_complaint` is the opposite - 13 of its 20
examples are minor rephrasings of the same "you used way too much packaging"
complaint about different products, and only ~3 are genuinely distinct root
issues (excess packaging, item damaged in transit, mislabeled gift item).
Rough hand-count across the full set: on the order of 90-100 distinct root
issues behind the 208 tweets, not 208 - noted honestly rather than claiming
208 independent data points.

## 5. Evaluation harness

- **Intent**: precision/recall/F1 per class + confusion matrix
  (`eval/metrics.py:intent_metrics`), for both the LLM classifier and the
  TF-IDF baseline. Classes are imbalanced by construction of the *real* data
  even though the golden set itself is balanced 26/26 - `delivery_problem`
  is ~3x the corpus volume of `prime_membership_value` - flagged, not
  ignored.
- **Reply quality**: LLM-as-judge (`eval/judge.py`) on 4 dimensions -
  grounded in precedent, tone, factual consistency, actionable - scored
  1-5. **Judge validated against a human**: 35 examples (seeded sample of
  the system's non-escalated replies) hand-scored on the same rubric,
  independently of the judge's own scores. Cohen's kappa (linear-weighted,
  since these are ordinal 1-5 scores) between human and judge:

  | dimension | kappa | reading |
  |---|---|---|
  | grounded | 0.582 | moderate |
  | tone | 0.470 | weak-moderate |
  | factual | 0.495 | weak-moderate |
  | actionable | 0.652 | substantial |

  Reported honestly: `tone` and `factual` are the weakest - the judge
  and I agreed more often on whether a reply gave the customer something to
  *do* (`actionable`) than on subtler calls like whether a claim was
  factually overreaching (`factual`) or whether phrasing read as
  professional (`tone`). Full per-example comparison in
  `golden_set/human_judge_agreement.csv`.
- **Escalation**: precision/recall against the hand labels, **and**
  cost-weighted, not just accuracy - a missed safety-relevant message costs
  10x a missed ordinary escalation, which costs 3x an unnecessary
  escalation (`eval/metrics.py:COST_*`). Stated explicitly because a single
  accuracy number would treat "sent an unnecessary human handoff" and
  "auto-replied to a self-harm mention" as the same kind of mistake.

## 6. What's misleading about my headline number

- **The escalation policy's cost number looks fine (0.596) until you check
  *why* it escalates - then it looks bad.** Every single one of the 107/208
  (51%) escalations the policy triggered came from exactly one rule:
  `classifier_disagreement` (baseline TF-IDF intent != LLM intent). Zero
  came from safety, legal/PR, refund-amount, repeated-contact, or
  low-precedent - those five triggers *never fired once* on the full 208-
  example set. Because the TF-IDF baseline (46.6% accurate) disagrees with
  the LLM classifier (68.8% accurate) constantly, "disagreement" alone fires
  on over half of all traffic - and its precision is 0.15 (recall 0.593).
  The policy's average cost (0.596/example) is actually *worse* than simply
  never escalating (0.389/example, `escalation_never` in
  `outputs/eval_report_full.json`) - the volume of unnecessary escalations
  (91 false positives, cost 1 each = 91) outweighs the value of the true
  positives it catches (16 of 27, cost saved vs. missing them). A
  reasonable-looking system built the escalation trigger that, on this
  measurement, makes things worse than doing nothing - see failure analysis
  #1 for the fix direction (don't use a weak baseline as one half of a
  disagreement signal). This was found only by breaking the cost number down
  by trigger, not by trusting it at face value - exactly the point of this
  section.
- **Judge and drafter share a model family.** Both default to
  `qwen2.5:7b-instruct`. A judge grading its own family's phrasing can be
  systematically lenient in ways a truly independent judge wouldn't be -
  this is exactly why the human-agreement kappa matters more than the raw
  judge score, and why that kappa is reported even if it's mediocre.
- **Small, self-labeled golden set.** 208 examples, labeled by one person
  (me) reading the text - not a second annotator, no adjudication. Genuinely
  ambiguous calls (see decision log #16 for several) went whichever way I
  read them at the time; a second labeler would disagree on some fraction of
  these, and that fraction isn't measured here.
- **Public tweets only, no account context.** Real AmazonHelp agents have
  order history, account status, prior contact history. Our "grounded"
  replies are grounded in *less* information than a real agent has - the
  retrieval precedent is the only context, not the actual account. A
  "correct" reply here is at most "as good as what the brand's own public
  tweets show," not "actually resolves the order."
- **Brand-specific, won't generalize.** The intent taxonomy, the escalation
  keyword lists, and the retrieval corpus are all AmazonHelp-specific. A
  brand with a different tone (e.g. SpotifyCares, much lower volume/depth in
  our own profiling) or a different issue mix would need a redone
  clustering pass, not just a config swap.
- **Time-boxed data, policy drift.** The corpus is a historical snapshot
  (2017-era tweets, see `created_at` in the raw data). Shipping SLAs, refund
  policies, and even the brand's reply tone may have changed since -
  precedent retrieval is grounded in what AmazonHelp *used to* do, not
  necessarily what it does now.
- **Local LLM, not the LLM a production system would use.** Everything here
  runs on a locally-hosted `qwen2.5:7b-instruct`/`14b-instruct` (no cloud API
  key was available in the build environment). Numbers - especially reply
  quality - are specific to this model's capability level and would shift,
  probably upward, with a larger frontier model.
- **Escalation regex gaps are real, not hypothetical.** Hand-labeling
  surfaced concrete cases the rule-based policy's keyword patterns miss
  (decision log #16) - abbreviated dollar amounts, "illegal" without the
  literal phrase "legal action," repeated-contact language outside the
  regex's specific patterns. These show up as false negatives in the
  escalation metrics, not as a hidden gap - but a wider real-world vocabulary
  would surface more of them than 208 examples can.

## 7. Failure analysis (top 5, with real examples and hypotheses)

1. **The escalation policy over-triggers on classifier disagreement, and
   under-triggers on everything else.** Confirmed in section 6 - 91 false
   positives from `classifier_disagreement` alone, and its 5 sibling
   triggers fired zero times combined. Concrete misses: *"I no longer care
   why I didn't receive my original order... you don't have any [replacement]
   - nor do you know when you will!"* (hand-labeled: escalate, "repeated
   unresolved order issue, no replacement stock") and *"Sigh. Another day,
   another blown @AmazonHelp delivery deadline"* (hand-labeled: escalate,
   "recurring, unresolved delivery-failure pattern") were both auto-handled
   with a generic apology-and-DM reply instead. Hypothesis: `low_precedent`
   and `repeated_contact` are the triggers that *should* have caught these,
   but the retrieval corpus (1200 threads) and regex patterns aren't broad
   enough - see decision log #15, #16. Fix direction: replace
   `classifier_disagreement` with a confidence threshold on the *LLM*
   classifier alone (it's the more accurate one), and broaden the
   repeated-contact regex.
2. **`general_inquiry_or_other` catch-all bleed.** Confirmed in the
   confusion matrix: 9 examples true-labeled `general_inquiry_or_other` were
   predicted `poor_support_experience`, 6 more predicted
   `prime_membership_value`. Example: *"Where do you declare the winner of
   any contest.I hv been following it, every time you organise contest and
   don't announce the winner"* - true intent `general_inquiry_or_other`
   (it's a contest question, not a real complaint), but the baseline
   classifier predicted `delivery_problem` (this specific disagreement with
   the LLM's correct call is also what triggered a false-positive escalation
   above). Hypothesis: this bucket is lexically inconsistent by
   construction (praise, noise, questions, off-topic mentions all live
   here), so nothing about "the words in the tweet" reliably signals it -
   a classifier trained mostly on the other 7 more lexically-coherent
   intents will default to whichever of those looks closest.
3. **`device_or_app_tech_support` -> `general_inquiry_or_other` confusion (6
   cases)** - the reverse direction of #2. Device/app messages that are
   *feature requests or praise* rather than *bug reports* ("when will we be
   able to play on alexa within the music app" type wording) read lexically
   like general feedback, and get pulled into the catch-all bucket.
4. **Low-diversity intents inflate apparent classifier accuracy.**
   `packaging_or_damage_complaint`'s ~3 real root issues restated across 20
   examples (section 4) makes it one of the easier classes to score well on
   without generalizing - contrast with `payment_or_giftcard_issue`, which
   still confuses with `refund_or_return_request` in 5 cases (*"I bought
   Season 2 of Riverdale but it's making me pay for the 6th episode"* -
   billing vs. refund framing is a genuinely fine line even for a human).
5. **langdetect false negatives let non-English content into an
   English-only pipeline.** At least two clearly non-English messages
   ("MEUS LIVROS CHEGARAM HOJE..." - Portuguese, "Fire TV買ったー" -
   Japanese) survived the English-only filter into the golden-set sample
   (flagged during hand-labeling, see `golden_set/golden_set.csv` rows 20
   and 76). `langdetect` is unreliable on short, emoji-heavy, code-mixed
   text - a production filter would need a stronger model or a
   length/confidence threshold before trusting its verdict.

## 8. What I'd do next with one more week

- Fix the escalation policy first, not last - section 6/7 #1 found it's
  currently worse than never escalating on this cost model, entirely because
  of one over-firing trigger. Swap `classifier_disagreement` for an
  LLM-confidence threshold and re-measure before touching anything else;
  this is the single highest-leverage change surfaced by this eval.
- Second human labeler + adjudication on the golden set, to actually measure
  labeling agreement instead of assuming my own reading is ground truth.
- A judge from a different model family than the drafter, to remove the
  correlated-error concern in section 6 instead of just flagging it.
- Expand the escalation policy's keyword patterns using the false negatives
  surfaced here as a starting set - not to fit these exact 208 examples, but
  as seeds for a broader pattern list, ideally validated against a *new*
  held-out sample so it isn't just overfit to this golden set.
- Try a larger/frontier LLM for drafting and re-run the same harness, to
  separate "the pipeline design is the bottleneck" from "the 7B local model
  is the bottleneck."
- Real order/account context (even mocked) to see how much of the "reply
  quality" ceiling in section 6 is actually about DM-based information, not
  about the tweet-reply layer at all.
