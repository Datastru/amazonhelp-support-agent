"""Ties classify -> retrieve -> draft -> escalate into one call. This is
"the system" - eval/run_eval.py calls this (and the baseline variants) over
the golden set."""
from pipeline.classify import baseline_classify, llm_classify
from pipeline.draft import draft_reply, template_baseline_reply
from pipeline.escalate import decide, keyword_baseline
from pipeline.intents import MAJORITY_INTENT


def run_agent(text, index, k=3):
    baseline_intent = baseline_classify(text)
    llm_intent = llm_classify(text)
    retrieved = index.retrieve(text, k=k)
    policy = decide(text, retrieved, baseline_intent=baseline_intent, llm_intent=llm_intent)

    reply = None
    if not policy["escalate"]:
        reply = draft_reply(text, retrieved)

    return {
        "text": text,
        "baseline_intent": baseline_intent,
        "llm_intent": llm_intent,
        "intent": llm_intent,  # the system's intent-of-record
        "retrieved": retrieved,
        "escalate": policy["escalate"],
        "escalate_trigger": policy["trigger"],
        "escalate_reason": policy["reason"],
        "reply": reply,
    }


def run_trivial_baseline(text, index=None):
    """Majority-class intent, canned reply, escalate nothing."""
    return {
        "text": text,
        "intent": MAJORITY_INTENT,
        "escalate": False,
        "escalate_trigger": "none",
        "escalate_reason": "trivial baseline never escalates",
        "reply": template_baseline_reply(text),
    }


def run_simple_baseline(text, index):
    """TF-IDF intent, template reply (no retrieval grounding), keyword escalation."""
    intent = baseline_classify(text)
    retrieved = index.retrieve(text, k=3)
    policy = keyword_baseline(text)
    reply = None if policy["escalate"] else template_baseline_reply(text)
    return {
        "text": text,
        "intent": intent,
        "retrieved": retrieved,
        "escalate": policy["escalate"],
        "escalate_trigger": policy["trigger"],
        "escalate_reason": policy["reason"],
        "reply": reply,
    }
