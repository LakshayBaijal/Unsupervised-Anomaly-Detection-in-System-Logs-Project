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
