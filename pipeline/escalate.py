"""
Rule-based escalation policy. Every decision carries a named reason, not just
a confidence float - "confidence < 0.7" doesn't tell a reviewer anything,
"safety keyword hit" or "no precedent found" does.

Checked in priority order, first match wins - a safety mention outranks
everything else regardless of what else is true about the message.
"""
import re

SAFETY_RE = re.compile(
    r"\b(kill myself|suicid|self.?harm|hurt myself|end my life|going to die|"
    r"gun|weapon|threat(en)?(ing)?\b.*\b(you|kill|hurt))",
    re.I,
)
LEGAL_PR_RE = re.compile(
    r"\b(lawyer|attorney|lawsuit|sue you|sue us|legal action|class action|"
    r"press|reporter|journalist|news station|better business bureau|\bftc\b|"
    r"discriminat)",
    re.I,
)
REPEATED_RE = re.compile(
    r"\b(third time|3rd time|again and again|still (not|hasn'?t)|"
    r"\d+ (days|times) in a row|every time i|for the (third|fourth|fifth) time)",
    re.I,
)
AMOUNT_RE = re.compile(r"\$\s?(\d{1,6}(?:\.\d{2})?)")

REFUND_THRESHOLD_USD = 100.0
LOW_PRECEDENT_SIM = 0.45


def _amount_over_threshold(text):
    amounts = [float(m) for m in AMOUNT_RE.findall(text)]
    return any(a >= REFUND_THRESHOLD_USD for a in amounts)


def decide(text, retrieved, baseline_intent=None, llm_intent=None):
    """retrieved: list of {..., "similarity": float} from Index.retrieve(),
    already sorted best-first. Returns {"escalate": bool, "trigger": str,
    "reason": str}."""
    if SAFETY_RE.search(text):
        return {"escalate": True, "trigger": "safety", "reason": "message mentions self-harm/harm - always route to a human"}

    if LEGAL_PR_RE.search(text):
        return {"escalate": True, "trigger": "legal_pr", "reason": "legal or PR-sensitive language detected"}

    if _amount_over_threshold(text):
        return {"escalate": True, "trigger": "high_value_refund", "reason": f"refund/compensation amount mentioned >= ${REFUND_THRESHOLD_USD:.0f}"}

    if REPEATED_RE.search(text):
        return {"escalate": True, "trigger": "repeated_contact", "reason": "customer signals this is a repeated, unresolved contact"}

    best_sim = retrieved[0]["similarity"] if retrieved else 0.0
    if best_sim < LOW_PRECEDENT_SIM:
        return {"escalate": True, "trigger": "low_precedent", "reason": f"no close precedent in resolved-thread index (best similarity {best_sim:.2f})"}

    if baseline_intent is not None and llm_intent is not None and baseline_intent != llm_intent:
        return {"escalate": True, "trigger": "classifier_disagreement", "reason": f"baseline says '{baseline_intent}', LLM says '{llm_intent}' - no consensus"}

    return {"escalate": False, "trigger": "none", "reason": "no escalation trigger fired, precedent found, classifiers agree"}


def keyword_baseline(text):
    """Simple baseline for comparison: escalate on a flat keyword list only,
    no precedent/agreement signals."""
    if SAFETY_RE.search(text) or LEGAL_PR_RE.search(text):
        return {"escalate": True, "trigger": "keyword", "reason": "safety or legal keyword matched"}
    return {"escalate": False, "trigger": "none", "reason": "no keyword matched"}
