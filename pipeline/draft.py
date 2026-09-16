"""Reply drafting: the retrieval-grounded version and a template baseline."""
import re

from pipeline.llm_client import chat

MENTION_RE = re.compile(r"^@\S+\s*")

DRAFT_SYSTEM = (
    "You are a support agent for Amazon's customer support Twitter account "
    "(@AmazonHelp). Write a short reply (1-3 sentences, under 280 characters) "
    "to the customer's tweet. Match the tone and structure of the example "
    "past replies you're given - they show how this brand actually resolves "
    "similar issues. Don't invent order numbers, refund amounts, or policies "
    "not implied by the examples. If the examples ask the customer to move to "
    "DMs for account details, do the same."
)


def draft_reply(text, retrieved, model=None):
    """retrieved: top-k precedent dicts from Index.retrieve() - issue_text +
    resolution_reply pairs used as few-shot grounding, not just style cues."""
    examples = "\n\n".join(
        f"Past similar issue: {r['issue_text']}\nPast reply that resolved it: {r['resolution_reply']}"
        for r in retrieved
    )
    prompt = f"{examples}\n\nNew customer tweet: {text}\n\nYour reply:"
    kwargs = {"system": DRAFT_SYSTEM}
    if model:
        kwargs["model"] = model
    return chat(prompt, **kwargs).strip()


TEMPLATE_BASELINE = (
    "Hi, thanks for reaching out - we're sorry for the trouble. "
    "Please send us a DM with your order number so we can look into this."
)


def template_baseline_reply(text, retrieved=None):
    """Trivial baseline: one canned reply regardless of content."""
    return TEMPLATE_BASELINE
