"""Thin wrapper over the local Ollama REST API. No SDK, just requests."""
import requests

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:7b-instruct"
CHAT_MODEL_LARGE = "qwen2.5:14b-instruct"
EMBED_MODEL = "nomic-embed-text"


def chat(prompt, system=None, model=CHAT_MODEL, temperature=0.2):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def embed(texts, model=EMBED_MODEL, batch_size=64):
    """Returns a list of embedding vectors, one per input text.

    Uses /api/embed (batched, takes a list) not /api/embeddings (one at a
    time) - the per-request overhead made single-item calls ~2s each, batching
    got that down to ~70ms/item.
    """
    if isinstance(texts, str):
        texts = [texts]
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": model, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        vecs.extend(resp.json()["embeddings"])
    return vecs
