"""
Pulls the TWCS (Twitter Customer Support) dataset.

Kaggle's thoughtvector/customer-support-on-twitter needs a Kaggle API token.
We don't have one in this env, so we use a public parquet mirror on the HF Hub
with the identical schema (tweet_id, author_id, inbound, created_at, text,
response_tweet_id, in_response_to_tweet_id) and row count (~2.81M) -
SunidhiSriram/twcs. No login required, plain HTTPS GET.
"""
import os
import urllib.request

PARQUET_URLS = [
    "https://huggingface.co/api/datasets/SunidhiSriram/twcs/parquet/default/train/0.parquet",
    "https://huggingface.co/api/datasets/SunidhiSriram/twcs/parquet/default/train/1.parquet",
]
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def fetch(url, out_path):
    if os.path.exists(out_path):
        print(f"already have {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB), skipping")
        return
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        last_reported = 0
        chunk = resp.read(1 << 20)
        while chunk:
            f.write(chunk)
            done += len(chunk)
            if total and done - last_reported > 10 << 20:
                print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB")
                last_reported = done
            chunk = resp.read(1 << 20)
    print()


def main():
    raw_dir = os.path.abspath(RAW_DIR)
    os.makedirs(raw_dir, exist_ok=True)
    for i, url in enumerate(PARQUET_URLS):
        fetch(url, os.path.join(raw_dir, f"twcs_{i}.parquet"))


if __name__ == "__main__":
    main()
