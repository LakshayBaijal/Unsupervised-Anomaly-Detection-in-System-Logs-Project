#!/usr/bin/env python3
# src/features.py

import os
import pickle
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import sys

def build_features(parsed_csv="data/parsed_logs.csv", out_prefix="data/features"):
    if not os.path.exists(parsed_csv):
        raise FileNotFoundError(f"{parsed_csv} not found. Run parser first.")

    df = pd.read_csv(parsed_csv)
    messages = df["message"].fillna("").astype(str).tolist()

    # --- WORD TF-IDF VECTORIZER ---
    word_vect = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 4),    # more expressive
        stop_words=None
    )
    X_word = word_vect.fit_transform(messages)

    # --- CHARACTER TF-IDF VECTORIZER ---
    char_vect = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 7),
        max_features=2000
    )
    X_char = char_vect.fit_transform(messages)

    # --- CONCATENATE INTO FINAL FEATURE MATRIX ---
    X = hstack([X_word, X_char], format="csr")

    # --- SAVE OUTPUT ---
    sp.save_npz(out_prefix + ".npz", X)

    with open(out_prefix + "_word_vectorizer.pkl", "wb") as f:
        pickle.dump(word_vect, f)

    with open(out_prefix + "_char_vectorizer.pkl", "wb") as f:
        pickle.dump(char_vect, f)

    print(f"Built features: shape = {X.shape}")
    print(f"Saved matrix to {out_prefix}.npz")
    print("Saved word + char vectorizers.")

    return X, df, (word_vect, char_vect)

if __name__ == "__main__":
    parsed_csv = sys.argv[1] if len(sys.argv) > 1 else "data/parsed_logs.csv"
    build_features(parsed_csv)
