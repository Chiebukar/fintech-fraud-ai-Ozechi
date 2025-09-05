# Fraud Detection in FinTech Transactions

## Project Overview

This project focuses on building an **AI-powered fraud detection system** for fintech transaction data. The pipeline spans **data preprocessing, feature engineering, imbalance handling, model development, explainability (SHAP), and deployment (API + Streamlit app)**. The aim is to build a production-grade solution that not only detects fraud but also provides interpretability and business insights.

---

## Data Understanding and Preprocessing

* Dataset: **1,000 synthetic fintech transactions** with 14 features (transaction details, customer IDs, device/location, and risk indicators).
* Data checks included:

  * No missing values, no duplicates
  * Temporal coverage: Jan 1, 2024 – Feb 11, 2024
  * Categorical variables encoded (`customer_id`, `transaction_type`, etc.)
  * Numerical variables standardized

---

## Feature Engineering Approach

* **Log Transformation**: Applied to `transaction_amount` for normalization.
* **Behavioral Features**:

  * `past_mean_amount`: deviation from historical non-fraud transaction averages.
  * `recency_ratio`: abnormal gaps between transactions.
  * `geo_switch_risk`, `device_switch_risk`, `transaction_type_switch`: detect sudden switches in device/location/transaction type.
* **Standardization**: Numerical features scaled with `StandardScaler`.
* **Encoding**: Ordinal encoding applied to categorical features.

---

## Model Selection and Training Strategy

1. **Baseline**: Dummy classifier (majority class) → Accuracy ≈ 78%, Recall for fraud = 0%.
2. **Model Candidates**: RandomForest, XGBoost, LightGBM, Gradient Boosting, ExtraTrees.
3. **Evaluation Metrics**: Precision, Recall, F1-score (chosen over accuracy due to class imbalance).
4. **Cross-validation**: Used **TimeSeriesSplit** to respect temporal ordering and avoid leakage.
5. **Final Model**: Gradient Boosting with **cost-sensitive learning** (class weights) + **noise downweighting** + **early stopping**.

---

## Temporal Analysis Results

* No clear impact of temporal difference on Fraud patterns.
* Device and location switches within **minutes** raise fraud likelihood.
* Temporal splits (80/20) ensured realistic training/testing.

---

## Imbalance Handling Strategy

* Fraud = **17%** of transactions.
* Approaches considered: resampling, synthetic oversampling (SMOTE), cost-sensitive learning.
* **Final Choice**: Cost-sensitive learning for stability and production simplicity.

---

## 🔍 Explainability Insights

Using **SHAP values**:

* Top drivers of fraud:

  * `risk_score` (higher = more fraud likelihood)
  * `log_transaction_amount` (large amounts correlate with fraud)
  * `is_foreign_transaction` (foreign = higher fraud probability)
* Behavioral anomalies (`recency_ratio`, `switch features`) also influenced predictions.
* Local SHAP plots provided **instance-level explanations** to justify alerts.

---

## Deployment Instructions

### Clone the Repository

```bash
git clone https://github.com/Chiebukar/AI_Egineer_Assessment.git
cd fraud-detection
```

### Train the Model

Train from scratch using the training utility:

```bash
python3 ./src/utils/train.py
```

### Score Transactions

Run the scoring script on new transaction data:

```bash
python3 ./src/utils/score.py
```

### 4Run the Flask API

Start the fraud detection API service:

```bash
python3 server.py
```

Endpoint available at:

```http
POST http://localhost:5000/score
```

Body: JSON transaction record → Returns fraud probability & class.

### 5️⃣ Run the Streamlit App

Launch the interactive fraud dashboard:

```bash
streamlit run streamlit_app.py
```

---

## 📌 Assumptions and Limitations

* Synthetic dataset (1,000 samples) may not reflect real-world fraud complexity.
* Temporal leakage risks are mitigated with proper splits, but real-time drift must be monitored.
* Model assumes features provided at transaction time — some external signals (e.g., IP data, network graph features) not included.

---

## Future Improvements

* Explore **Autoencoder anomaly detection** for temporal fraud sequences.
* Integrate **active learning feedback loop** from fraud analysts.
* Expand dataset and evaluate with **real-world transactions**.

---
