#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.ensemble import IsolationForest

import sys
import os

from features import build_features

def extract_anomalies(parsed_csv="data/parsed_logs.csv",
                      features_file="data/features.npz",
                      word_vec_file="data/features_word_vectorizer.pkl",
                      char_vec_file="data/features_char_vectorizer.pkl",
                      model_file="data/isoforest_model.pkl",
                      out_file="data/anomalies_extracted.csv",
                      top_k=50):
    
    # Load parsed logs
    df = pd.read_csv(parsed_csv)
    messages = df["message"].fillna("").astype(str).tolist()

    # Load vectorizers
    with open(word_vec_file, "rb") as f:
        word_vec = pickle.load(f)
    with open(char_vec_file, "rb") as f:
        char_vec = pickle.load(f)

    # Transform messages using saved vectorizers
    X_word = word_vec.transform(messages)
    X_char = char_vec.transform(messages)

    X = sp.hstack([X_word, X_char], format="csr")

    # Load trained model
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Get anomaly scores (lower = more anomalous)
    scores = model.decision_function(X)
    
    df["anomaly_score"] = scores
    df["is_anomaly"] = (scores < np.percentile(scores, 5)).astype(int)  # top 5% anomalies

    # Extract top anomalies
    anomalies = df.sort_values("anomaly_score").head(top_k)

    anomalies.to_csv(out_file, index=False)
    print(f"Saved top {top_k} anomalies to {out_file}")
    print(anomalies[["timestamp", "service", "message", "anomaly_score"]].head(10))

if __name__ == "__main__":
    extract_anomalies()
