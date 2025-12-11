#!/usr/bin/env python3
# src/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Log Anomaly Detection Dashboard", layout="wide")

# -------------------------------
# Load Data + Models
# -------------------------------

@st.cache_data
def load_logs():
    df = pd.read_csv("data/parsed_logs.csv")
    return df

@st.cache_data
def load_features():
    X = load_npz("data/features.npz")
    return X

@st.cache_data
def load_model():
    with open("data/isoforest_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

df = load_logs()
X = load_features()
model = load_model()

st.title("🔍 Log Anomaly Detection Dashboard")
st.markdown("A real-time visualization dashboard for anomaly detection in Linux system logs using **Isolation Forest + TF-IDF**.")

# -------------------------------
# Compute anomaly scores
# -------------------------------

scores = model.decision_function(X)
threshold = np.percentile(scores, 2)  # bottom 2% treated as anomalies
pred = (scores < threshold).astype(int)

df["anomaly_score"] = scores
df["is_anomaly"] = pred


# -------------------------------
# Sidebar controls
# -------------------------------

st.sidebar.header("⚙️ Controls")
score_cutoff = st.sidebar.slider("Anomaly Threshold Percentile", 1, 10, 2)

threshold = np.percentile(scores, score_cutoff)
df["is_anomaly"] = (scores < threshold).astype(int)

st.sidebar.write("Detected anomalies:", df["is_anomaly"].sum())


# -------------------------------
# Section 1 — Time-series chart
# -------------------------------

st.subheader("📈 Anomaly Score Over Log Sequence")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(scores, alpha=0.6, label="Score")
ax.scatter(
    np.where(df["is_anomaly"] == 1),
    scores[df["is_anomaly"] == 1],
    color="red",
    s=15,
    label="Anomalies"
)
ax.set_title("Anomaly Score Timeline")
ax.set_xlabel("Log Index")
ax.set_ylabel("Anomaly Score")
ax.legend()

st.pyplot(fig)


# -------------------------------
# Section 2 — Show Anomaly Table
# -------------------------------

st.subheader("🚨 Detected Anomalies (Filtered Table)")

anomalies = df[df["is_anomaly"] == 1][[
    "timestamp",
    "service",
    "message",
    "anomaly_score"
]]

st.dataframe(anomalies, height=300)


# -------------------------------
# Section 3 — Download CSV
# -------------------------------

csv_out = anomalies.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Anomaly Report (CSV)",
    data=csv_out,
    file_name="detected_anomalies.csv",
    mime="text/csv"
)


# -------------------------------
# Section 4 — Search Logs
# -------------------------------

st.subheader("🔎 Search Logs")

keyword = st.text_input("Enter keyword (e.g., 'ssh', 'error', 'fail'):")

if keyword.strip() != "":
    results = df[df["message"].str.contains(keyword, case=False, na=False)]
    st.write(f"Found {len(results)} matching logs:")
    st.dataframe(results, height=200)
