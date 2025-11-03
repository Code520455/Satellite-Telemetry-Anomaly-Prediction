# Satellite-Telemetry-Anomaly-Prediction

## Overview
This project focuses on detecting anomalies in satellite telemetry data using machine learning models.  
By analyzing sensor readings (`segments.csv`) and engineered features (`dataset.csv`), the system learns to classify whether a given telemetry segment is **normal** or **anomalous**.  

The solution leverages **Python**, **Pandas**, **Scikit-Learn**, **XGBoost**, **LightGBM** for data preprocessing, exploratory data analysis (EDA), and predictive modeling.

---

##  Project Objectives
- Load and preprocess raw satellite telemetry data.
- Engineer meaningful statistical and signal-based features.
- Perform Exploratory Data Analysis (EDA) to visualize patterns and anomalies.
- Train and evaluate machine learning models for anomaly classification.
- Compare performance metrics (Accuracy, ROC-AUC, Confusion Matrix).

---

## Dataset Description

### 1. `segments.csv`
Contains raw telemetry readings with timestamps and anomaly flags.

| Column | Description |
|--------|-------------|
| `channel` | Unique identifier of the telemetry channel |
| `timestamp` | UTC timestamp of each reading |
| `value` | Recorded sensor value |
| `label` | Indicates `anomaly` or `normal` |
| `sampling` | Sampling rate indicator |
| `anomaly` | 1 for anomaly, 0 for normal |
| `segment` | Segment ID grouping readings |
| `train` | Indicates if the sample belongs to training data |

### 2. `dataset.csv`
Contains engineered segment-level features for modeling.

| Column | Description |
|--------|-------------|
| `segment` | Segment ID |
| `anomaly` | Target variable (1 = anomaly, 0 = normal) |
| `mean`, `var`, `std`, `skew`, `kurtosis` | Statistical features |
| `n_peaks`, `diff_peaks`, `diff_var`, etc. | Signal-based features |
| `var_div_len`, `var_div_duration` | Normalized variance ratios |
| `len_weighted`, `gaps_squared` | Derived time-based metrics |

---

## Machine Learning Pipeline

### 1. Data Preprocessing
- Convert timestamps to datetime.
- Handle missing values (none found).
- Normalize numeric features if needed.
- Split dataset into training and test sets.

### 2. Exploratory Data Analysis (EDA)
- Distribution of anomalies vs. normal readings.
- Correlation heatmaps for feature relationships.
- Trend visualization using `matplotlib` and `seaborn`.

### 3. Model Training
Models used include:
- **Random Forest Classifier**
- **XGBoost**


### 4. Evaluation Metrics
- **Accuracy Score**
- **Confusion Matrix**
- **ROC Curve and AUC**

---

##  Requirements

### Python Libraries
Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost tensorflow jupyterlab


## Result
- From the analysis I found that there is a high correlation in these features Variance(var), peaks(diff_peaks,   smooth20_n_peaks) and higher order difference (diff2_var) with anomalies.
- From the confusion matrix it detects that Nomal: 251 True negatives correctly identified normal
  samples and 1 False positive.
- Also for Anomaly: 13 false negatives which means missed 13 anomalies and 54 True positives were correctly        identified   as anomalies
