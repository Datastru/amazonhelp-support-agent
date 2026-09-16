"""LLM-as-judge for reply quality. Scores 1-5 on four dimensions from the
brief's own rubric: grounded in precedent, correct tone, factually
consistent, actionable. Validated against a human (dev) hand-score pass in
golden_set/human_judge_agreement.csv - see eval/metrics.py for the kappa."""
import json

from pipeline.llm_client import chat

JUDGE_SYSTEM = (
    "You are grading a customer-support reply for Amazon's Twitter support "
    "account. Score it 1-5 on each dimension:\n"
    "grounded: consistent with the past resolution example(s) given, not "
    "inventing new policy\n"
    "tone: matches a helpful, professional support-agent tone\n"
    "factual: doesn't make claims/promises not supported by the tweet or the precedent\n"
    "actionable: customer knows what will happen next or what to do next\n"
    "Reply with JSON only: "
    '{"grounded": 1-5, "tone": 1-5, "factual": 1-5, "actionable": 1-5}'
)


def judge_reply(customer_text, reply, retrieved, model=None):
    precedent = "\n".join(f"- {r['resolution_reply']}" for r in retrieved) or "(none retrieved)"
    prompt = (
        f"Customer tweet: {customer_text}\n\n"
        f"Past precedent replies:\n{precedent}\n\n"
        f"Reply being graded: {reply}\n\nJSON:"
    )
    kwargs = {"system": JUDGE_SYSTEM}
    if model:
        kwargs["model"] = model
    raw = chat(prompt, **kwargs)
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        scores = json.loads(raw[start:end])
        return {k: int(scores[k]) for k in ("grounded", "tone", "factual", "actionable")}
    except (ValueError, KeyError):
        return {"grounded": None, "tone": None, "factual": None, "actionable": None}
