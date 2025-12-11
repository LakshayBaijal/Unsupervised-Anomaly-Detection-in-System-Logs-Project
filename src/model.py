#!/usr/bin/env python3
# src/model.py

import pickle
import scipy.sparse as sp
from sklearn.ensemble import IsolationForest
import sys
import os

def train_model(features_npz="data/features.npz",
                model_out="data/isoforest_model.pkl",
                contamination=0.06,
                n_estimators=300):

    if not os.path.exists(features_npz):
        raise FileNotFoundError(f"{features_npz} not found. Run features.py first.")

    X = sp.load_npz(features_npz)

    # Create Isolation Forest (modern sklearn, no 'behaviour' arg)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42
    )

    # Try sparse fit, fallback to dense if needed
    try:
        model.fit(X)
    except Exception:
        print("Sparse fit failed — converting to dense. (May be slower)")
        model.fit(X.toarray())

    with open(model_out, "wb") as f:
        pickle.dump(model, f)

    print(f"Isolation Forest trained and saved to {model_out}")
    return model

if __name__ == "__main__":
    features_file = sys.argv[1] if len(sys.argv) > 1 else "data/features.npz"
    train_model(features_file)
