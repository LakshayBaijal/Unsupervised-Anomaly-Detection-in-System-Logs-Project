#!/usr/bin/env python3
# src/evaluate.py

import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import random
import sys
import os

def inject_anomalies(df, fraction=0.02, seed=42, strong=True):
    random.seed(seed)
    n = len(df)
    k = max(1, int(n * fraction))
    indices = random.sample(range(n), k)
    df_copy = df.copy()
    for idx in indices:
        orig = df_copy.at[idx, "message"]
        orig_str = "" if pd.isna(orig) else str(orig)
        if strong:
            new_msg = ("CRITICAL_ERROR!!! SECURITY_BREACH!!! "
           "UNUSUAL_PATTERN_DETECTED!!! "
           "###ANOMALY### " + orig_str)

        else:
            new_msg = "ANOMALY_TRIGGER_ " + orig_str
        df_copy.at[idx, "message"] = new_msg
    labels = np.zeros(n, dtype=int)
    labels[indices] = 1
    return df_copy, labels

def vectorize_messages(df_messages, word_vect_file="data/features_word_vectorizer.pkl", char_vect_file="data/features_char_vectorizer.pkl"):
    with open(word_vect_file, "rb") as f:
        word_vect = pickle.load(f)
    with open(char_vect_file, "rb") as f:
        char_vect = pickle.load(f)
    Xw = word_vect.transform(df_messages)
    Xc = char_vect.transform(df_messages)
    X = sp.hstack([Xw, Xc], format="csr")
    return X

def evaluate(parsed_csv="data/parsed_logs.csv",
             word_vect_file="data/features_word_vectorizer.pkl",
             char_vect_file="data/features_char_vectorizer.pkl",
             model_file="data/isoforest_model.pkl",
             fraction=0.02):
    if not os.path.exists(parsed_csv):
        raise FileNotFoundError("parsed logs not found")

    df = pd.read_csv(parsed_csv)

    df_injected, true_labels = inject_anomalies(df, fraction=fraction, seed=42, strong=True)

    # Vectorize injected messages
    X = vectorize_messages(df_injected["message"].astype(str).tolist(), word_vect_file, char_vect_file)

    # Load model
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Score & predict
    try:
        scores = model.decision_function(X)
        preds = model.predict(X)
    except Exception:
        # fallback to dense
        arr = X.toarray()
        scores = model.decision_function(arr)
        preds = model.predict(arr)

    pred_labels = (preds == -1).astype(int)

    # Metrics (safe)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    try:
        auc = roc_auc_score(true_labels, -scores)
    except Exception:
        auc = float("nan")

    print("\n=== Evaluation Results ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"AUC:       {auc:.3f}")

    df_out = df.copy()
    df_out["anomaly_score"] = -scores
    df_out["predicted_anomaly"] = pred_labels
    df_out.to_csv("data/eval_results.csv", index=False)
    print("Saved data/eval_results.csv")

    return df_out

if __name__ == "__main__":
    evaluate()
