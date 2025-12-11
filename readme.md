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
