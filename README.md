Project: Unsupervised Anomaly Detection in System Logs

https://github.com/user-attachments/assets/b0695bc5-515a-4559-b523-62ad2c3bc502




Python · Scikit-learn · TF-IDF · Isolation Forest · NLP · Data Visualization

- Built an unsupervised anomaly detection system for Linux system logs using TF-IDF embeddings and Isolation Forest to automatically flag rare or suspicious events.
- Designed a custom log parser to extract timestamps, severity levels, and message patterns from syslog entries.
- Injected synthetic anomalies and achieved ~91% detection accuracy, demonstrating robustness across varied log types.
- Visualized anomaly scores and clustered log behavior using PCA and Matplotlib for interpretable security insights.

- Run the Dynamic Parser

```br
python3 src/parser.py data/Linux_2k.log
```

- Build TF-IDF Features

```br
python3 src/features.py data/parsed_logs.csv
```

- Train The Isolation Forest Model

```br
python3 src/model.py data/features.npz
```

- Evaluate Synthetic Anomaly

```br
python3 src/evaluate.py
```

- Visualizations

```br
python3 src/visualize.py
```

- Run the Dashboard

```br
streamlit run src/dashboard.py
```

## Metrics
<img width="275" height="138" alt="Screenshot from 2025-12-13 00-59-09" src="https://github.com/user-attachments/assets/13b274bd-fda9-4f3b-97b0-b5a0f78d3188" />

## Plots
<img width="1271" height="442" alt="anomaly_timeseries" src="https://github.com/user-attachments/assets/fafe35f4-ad3a-48ba-9e36-890b14ebf047" />

<img width="891" height="789" alt="pca" src="https://github.com/user-attachments/assets/887e7bb8-dcb4-4099-ba03-8d967802c76a" />
