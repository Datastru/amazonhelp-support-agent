"""
TF-IDF + LogisticRegression baseline intent classifier. Trained on the
cluster-derived silver labels from scripts/cluster_intents.py (the cluster
assignment IS the label - no separate hand-labeling round needed to get a
few hundred training rows, and no LLM calls either).
"""
import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="artifacts/training_labels.csv")
    ap.add_argument("--intent-map", default=None, help="cluster_id -> intent name csv, optional")
    args = ap.parse_args()

    df = pd.read_csv(args.labels)

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, stop_words="english")
    X = vec.fit_transform(df["text"])
    y = df["cluster_id"].astype(str)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pred = cross_val_predict(clf, X, y, cv=5)
    print("5-fold CV report (cluster id as label):")
    print(classification_report(y, pred, zero_division=0))

    clf.fit(X, y)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, "baseline_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
    print(f"saved model + vectorizer to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
