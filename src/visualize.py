#!/usr/bin/env python3
# src/visualize.py

import pandas as pd
import numpy as np
import os
import pickle
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import sys

def plot_svd(features_npz="data/features.npz", eval_csv="data/eval_results.csv", out_prefix="plots/pca"):
    if not os.path.exists(eval_csv):
        raise FileNotFoundError(f"{eval_csv} not found. Run evaluate.py first.")
    df = pd.read_csv(eval_csv)
    X = sp.load_npz(features_npz)

    # use TruncatedSVD for sparse data
    svd = TruncatedSVD(n_components=2, random_state=42)
    try:
        Xd = svd.fit_transform(X)
    except Exception:
        Xd = svd.fit_transform(X.toarray())

    colors = ['blue' if a==0 else 'red' for a in df['predicted_anomaly']]
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(7,6))
    plt.scatter(Xd[:,0], Xd[:,1], c=colors, s=8, alpha=0.7)
    plt.title('2D SVD of Log Features (red = predicted anomaly)')
    plt.savefig(out_prefix + '.png', dpi=150, bbox_inches='tight')
    print(f"Saved {out_prefix}.png")

def plot_timeseries(eval_csv="data/eval_results.csv", out="plots/anomaly_timeseries.png"):
    df = pd.read_csv(eval_csv)
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10,3))
    plt.plot(df['anomaly_score'].values, linewidth=0.8)
    if 'predicted_anomaly' in df.columns:
        idx = np.where(df['predicted_anomaly']==1)[0]
        plt.scatter(idx, df.loc[df['predicted_anomaly']==1, 'anomaly_score'], color='red', s=12)
    plt.title('Anomaly Score over time (red dots = predicted anomaly)')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")

if __name__ == "__main__":
    plot_svd()
    plot_timeseries()
