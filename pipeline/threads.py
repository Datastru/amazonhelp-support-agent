"""
Shared thread-reconstruction logic. Raw TWCS rows are a flat table of tweets
linked by response_tweet_id / in_response_to_tweet_id - this turns that into
actual conversation chains. Used by both scripts/profile_brands.py (which
brand to pick) and scripts/build_threads.py (build the chosen brand's data).
"""
import re

import pandas as pd
from langdetect import detect, LangDetectException

RAW_FILES = ["data/raw/twcs_0.parquet", "data/raw/twcs_1.parquet"]
MAX_TURNS = 20  # these threads don't run longer than this in practice

CLOSING_RE = re.compile(
    r"\b(glad|happy to help|welcome|no problem|take care|reach out|"
    r"anything else|let us know|hope this helps)\b",
    re.I,
)


def load_data():
    return pd.concat([pd.read_parquet(f) for f in RAW_FILES], ignore_index=True)


def index_corpus(df):
    """Build id -> field dicts once, then walk each root tweet's reply chain.

    Plain dicts, not df.loc - row-by-row pandas indexing is the bottleneck
    at this row count.
    """
    ids = df["tweet_id"].to_numpy()
    next_id = dict(zip(ids, df["response_tweet_id"]))
    author = dict(zip(ids, df["author_id"]))
    text = dict(zip(ids, df["text"]))
    inbound = dict(zip(ids, df["inbound"]))
    created_at = dict(zip(ids, df["created_at"]))
    id_set = set(ids)

    in_response_to = df["in_response_to_tweet_id"]
    is_root = df["inbound"] & (in_response_to.isna() | ~in_response_to.isin(id_set))
    roots = df.loc[is_root, "tweet_id"].tolist()

    chains = []
    for root_id in roots:
        chain = [root_id]
        cur = root_id
        for _ in range(MAX_TURNS):
            nxt = next_id.get(cur)
            if nxt is None or pd.isna(nxt):
                break
            nxt_id = int(str(nxt).split(",")[0])
            if nxt_id not in id_set:
                break
            cur = nxt_id
            chain.append(cur)
        chains.append(chain)

    fields = {"author": author, "text": text, "inbound": inbound, "created_at": created_at}
    return chains, fields


def thread_touches_brand(chain, fields, brand):
    return any(fields["author"][t] == brand for t in chain)


def is_resolved(chain, fields, brand):
    last = chain[-1]
    return fields["author"][last] == brand and bool(CLOSING_RE.search(str(fields["text"][last])))


MENTION_RE = re.compile(r"@\S+")


def is_english_thread(chain, fields):
    """First customer turn decides the thread's language. langdetect is
    unreliable on very short strings, so this is a heuristic, not ground
    truth - noted as a scope decision (English-only) in the report."""
    first_customer = next((t for t in chain if fields["inbound"][t]), None)
    if first_customer is None:
        return False
    text = MENTION_RE.sub("", str(fields["text"][first_customer])).strip()
    if len(text) < 8:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False
