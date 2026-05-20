# Unsupervised Anomaly Detection in System Logs

This project implements an end-to-end pipeline for detecting anomalies in system logs (like Linux syslogs) using unsupervised machine learning. It parses semi-structured log data, extracts robust text and meta-features, and isolates anomalous log activities using an Isolation Forest model. Finally, it provides an interactive Streamlit dashboard for real-time visualization and exploration of the anomalies.

## 🚀 Project Pipeline & Architecture

The system follows a modular data pipeline design:

1. **Log Parsing (`src/parser.py`)**: Reads raw `.log` files and uses dynamic regex/grok patterns to structure them into tabular format (timestamp, component, message).
2. **Feature Engineering (`src/features.py`)**: 
   - Extracts numerical features using robust **Hashing Vectorizer** (to handle Out-Of-Vocabulary words seamlessly) and **Character-level TF-IDF**.
   - Extracts **Meta Features** (e.g., log length, uppercase ratio, punctuation count) to catch structural anomalies.
   - Applies **TruncatedSVD** to reduce high-dimensional text data into dense, meaningful principal components.
3. **Model Training (`src/model.py`)**: Trains an unsupervised **Isolation Forest** model to detect outliers without relying on labeled data.
4. **Evaluation (`src/evaluate.py`)**: Synthetically injects anomalies to rigorously test and evaluate the model's performance on heavily imbalanced datasets.
5. **Dashboard & Visualization (`src/dashboard.py` / `src/visualize.py`)**: Provides interactive plotting and insights into anomaly scores, distributions, and log groupings.

## 📂 Repository Structure

```text
.
├── data/               # Raw logs, parsed CSVs, engineered features, and model artifacts
├── plots/              # Output directory for generated anomaly visualizations
├── src/
│   ├── parser.py       # Log parsing script
│   ├── features.py     # Feature extraction (Hashing, TF-IDF, Meta, SVD)
│   ├── model.py        # Isolation forest training workflow
│   ├── evaluate.py     # Synthetic anomaly injection & metrics computation
│   ├── visualize.py    # Static plot generation
│   └── dashboard.py    # Streamlit interactive dashboard
├── requirements.txt    # Python dependencies
└── readme.md           # Project documentation
```

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Unsupervised-Anomaly-Detection-in-System-Logs-Project.git
   cd Unsupervised-Anomaly-Detection-in-System-Logs-Project
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix/macOS:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

Execute the pipeline in the following order:

**1. Run the Dynamic Parser**
```bash
python src/parser.py data/Linux_2k.log
```

**2. Build Features (Hashing, TF-IDF & Meta Features)**
```bash
python src/features.py data/parsed_logs.csv
```

**3. Train The Isolation Forest Model**
```bash
python src/model.py data/features.npz
```

**4. Evaluate the Model (with Synthetic Anomalies)**
```bash
python src/evaluate.py
```

**5. Visualizations**
```bash
python src/visualize.py
```

**6. Run the Dashboard**
```bash
streamlit run src/dashboard.py
```

## Evaluation Metrics & Benchmarks

Our model has been rigorously evaluated using an extended suite of metrics suited for highly imbalanced anomaly detection. Below are the latest benchmark results and a detailed explanation of what each metric means and how it is calculated.

### 1. Results Summary

```text
=== Evaluation Results ===
Precision:   0.2667
Recall:      1.0000
F1 Score:    0.4211
F2 Score:    0.6452
ROC AUC:     0.9920
PR AUC:      0.7535

=== Extended Benchmark Metrics ===
Accuracy:          0.9450
Balanced Accuracy: 0.9719
Specificity:       0.9439
False Alarm (FPR): 0.0561
Miss Rate (FNR):   0.0000
NPV:               1.0000
Cohen's Kappa:     0.4022
MCC:               0.5017
Brier Score:       0.0970

=== Confusion Matrix ===
True Positives  (TP): 40
True Negatives  (TN): 1850
False Positives (FP): 110
False Negatives (FN): 0

=== Performance Benchmarks ===
Total Samples:       2000
Total Inference Time:0.1031 seconds
Latency per Sample:  0.0516 ms
Peak Memory Usage:   0.5949 MB
```

### 2. Standard Evaluation Metrics

*   **Precision:** Represents the proportion of predicted anomalies that were actually anomalies. 
    *   **Formula:** $Precision = \frac{TP}{TP + FP}$
*   **Recall (Sensitivity / TPR):** Represents the proportion of actual anomalies that were correctly detected. A Recall of 1.00 means the model successfully found 100% of all injected anomalies.
    *   **Formula:** $Recall = \frac{TP}{TP + FN}$
*   **F1 Score:** The harmonic mean of Precision and Recall. It balances the trade-off when classes are imbalanced.
    *   **Formula:** $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
*   **F2 Score:** Similar to F1, but weights Recall higher than Precision. In anomaly detection, catching anomalies (Recall) is usually more critical than avoiding false alarms.
    *   **Formula:** $F2 = (1 + 2^2) \times \frac{Precision \times Recall}{(2^2 \times Precision) + Recall}$
*   **ROC AUC (Receiver Operating Characteristic - Area Under Curve):** Measures the trade-off between True Positive Rate and False Positive Rate across all thresholds. An AUC of 0.9920 indicates excellent separability.
*   **PR AUC (Precision-Recall Area Under Curve):** More robust than ROC AUC for heavily imbalanced datasets. It evaluates the model strictly on its ability to handle the rare anomaly class.

### 3. Extended Statistical Benchmark Metrics

*   **Accuracy:** Overall percentage of correct predictions regardless of class. Notoriously misleading for imbalanced data, but provided for completeness.
    *   **Formula:** $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
*   **Balanced Accuracy:** Averages the true positive rate (Recall) and the true negative rate (Specificity). Outstanding for checking if the model is unbiased.
    *   **Formula:** $Balanced\ Accuracy = \frac{Recall + Specificity}{2}$
*   **Specificity (True Negative Rate):** The model's ability to correctly identify normal baseline logs. 
    *   **Formula:** $Specificity = \frac{TN}{TN + FP}$
*   **False Alarm (FPR - False Positive Rate):** The proportion of normal logs wrongly categorized as anomalies.
    *   **Formula:** $FPR = 1 - Specificity = \frac{FP}{FP + TN}$
*   **Miss Rate (FNR - False Negative Rate):** The most critical metric for security. It denotes the percentage of anomalies that slipped through undetected. (Currently 0.00%!).
    *   **Formula:** $FNR = 1 - Recall = \frac{FN}{FN + TP}$
*   **NPV (Negative Predictive Value):** The probability that a log predicted as "normal" is truly normal. 
    *   **Formula:** $NPV = \frac{TN}{TN + FN}$
*   **Cohen's Kappa:** Measures inter-rater reliability, taking into account the possibility of the agreement occurring by chance.
    *   **Formula:** $\kappa = \frac{p_o - p_e}{1 - p_e}$ *(where $p_o$ is observed agreement, $p_e$ is expected agreement by chance)*
*   **MCC (Matthews Correlation Coefficient):** Generally regarded as the best single-value metric for binary classification on imbalanced sets. A higher MCC indicates a highly robust model.
    *   **Formula:** $MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$
*   **Brier Score:** Evaluates the mean squared difference between predicted probability scores and the actual labels. Lower is better.
    *   **Formula:** $Brier = \frac{1}{N} \sum (predicted\_prob - actual)^2$

### 4. Confusion Matrix Terminology

*   **True Positives (TP):** Anomalies correctly flagged as anomalies.
*   **True Negatives (TN):** Normal logs correctly flagged as normal.
*   **False Positives (FP):** Normal logs incorrectly flagged as anomalies (False Alarms).
*   **False Negatives (FN):** Anomalies incorrectly flagged as normal logs (Misses).
