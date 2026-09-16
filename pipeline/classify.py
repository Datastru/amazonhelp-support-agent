"""Two intent classifiers, compared head to head in eval:
  baseline_classify: TF-IDF + LogisticRegression, trained on cluster-derived
                      silver labels (scripts/train_baseline_classifier.py)
  llm_classify:       few-shot prompted qwen2.5:7b, given the named taxonomy
"""
import json
import os
import pickle

from pipeline.intents import INTENTS, CLUSTER_TO_INTENT
from pipeline.llm_client import chat

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

_baseline_model = None
_baseline_vectorizer = None


def _load_baseline():
    global _baseline_model, _baseline_vectorizer
    if _baseline_model is None:
        with open(os.path.join(ARTIFACT_DIR, "baseline_classifier.pkl"), "rb") as f:
            _baseline_model = pickle.load(f)
        with open(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            _baseline_vectorizer = pickle.load(f)
    return _baseline_model, _baseline_vectorizer


def baseline_classify(text):
    model, vec = _load_baseline()
    X = vec.transform([text])
    cluster_id = model.predict(X)[0]
    return CLUSTER_TO_INTENT.get(str(cluster_id), "unclear")


CLASSIFY_SYSTEM = (
    "You classify a customer support tweet into exactly one intent from a "
    "fixed list. Reply with JSON only: {\"intent\": \"<one of the listed "
    "names>\"}. No other text."
)


def _taxonomy_block():
    return "\n".join(f"- {name}: {desc}" for name, desc in INTENTS.items())


def llm_classify(text, model=None):
    prompt = f"Intents:\n{_taxonomy_block()}\n\nTweet: {text}\n\nJSON:"
    kwargs = {"system": CLASSIFY_SYSTEM}
    if model:
        kwargs["model"] = model
    raw = chat(prompt, **kwargs)
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        intent = json.loads(raw[start:end])["intent"]
    except (ValueError, KeyError):
        return "unclear"
    return intent if intent in INTENTS else "unclear"
