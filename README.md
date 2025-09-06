# [Fraud Detection in Transactions](https://fintech-fraud-ai-ozechi-0.streamlit.app/)

## Project Overview

This project focuses on building an **AI-powered fraud detection system** for fintech transaction data. The pipeline utilizes **data preprocessing, feature engineering, imbalance handling, model development, explainability (SHAP), and deployment (API + Streamlit app)**. The aim is to provide a production-grade solution that detects fraudulent transactions. See the web app here: [Fintech-Fraud-Detection-App](https://fintech-fraud-ai-ozechi-0.streamlit.app/)

---

## Data Understanding and Preprocessing

* Dataset: **1,000 transactions** with 14 features (transaction details, customer IDs, device/location, and risk indicators).
* Data checks included:

  * No missing values, no duplicates
  * Temporal coverage: Jan 1, 2024 – Feb 11, 2024
  * Categorical variables encoded 
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

## Analysis Results

* Risk score, transaction amount, and foreign transaction are major predictors of Fraud.
* No clear impact of temporal difference on Fraud patterns.
* Device and location switches contribute little to fraud likelihood.
* Temporal splits (80/20) ensured realistic training/testing (avoid data leakages).

---

## Imbalance Handling Strategy

* Fraud = **17%** of transactions.
* Approaches considered: resampling, synthetic oversampling (SMOTE), cost-sensitive learning.
* **Final Choice**: Cost-sensitive learning for stability and production simplicity.

---

## Explainability Insights

Using **SHAP values**:

* Top drivers of fraud:

  * `risk_score` (higher = more fraud likelihood)
  * `log_transaction_amount` (large amounts correlate with fraud)
  * `is_foreign_transaction` (foreign = higher fraud probability)
* Behavioral anomalies (`recency_ratio`, `switch features`) also influenced predictions.
* Local SHAP plots provided **instance-level explanations** to justify alerts.
* Explainability confirms insights from exploratory analysis

---

## Prerequisites

- Python **3.9**, **3.10**, or **3.11**  
- Git is installed on your system  
- Recommended: virtual environment (venv or conda) to isolate dependencies

## Instructions

### Clone the Repository

```bash
git clone https://github.com/Chiebukar/fintech-fraud-ai-Ozechi.git
cd fintech-fraud-ai-Ozechi
```
### Setup Environment

Before training the model, create a new virtual environment and install dependencies from `requirements.txt`.

**Linux / MacOS:**
```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

### Train the Model

Train from scratch using the training utility:

```bash
python3 ./src/utils/train.py
```

### Score Transactions

Run the predict script on new transaction data:

```bash
python3 ./src/utils/predict.py --input ./src/data/new_sample.xlsx
```
Replace `./src/data/new_sample.xlsx` with path to excel file of new transaction data

### Run the Flask API

Start the fraud detection API service:

```bash
python3 server.py
```

Endpoint available at:

```http
POST http://localhost:5000/predict
```

Body: JSON transaction record → Returns fraud probability & class.

Example request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "transaction_id": "T123",
        "customer_id": "CUST1092",
        "transaction_amount": 200,
        "transaction_time": "2025-09-04 12:30:00",
        "device_type": "Mobile",
        "location": "Lagos",
        "transaction_type": "Transfer",
        "is_foreign_transaction": 0,
        "previous_fraud_flag": 0,
        "is_high_risk_country": 0,
        "day_of_week": "Fri",
        "time_of_day": "Afternoon",
        "risk_score": 2.5
      }'
```
or from another python service using the requests library

```python
import requests

# Flask API endpoint
url = "http://localhost:5000/predict"

# Example transaction payload
transaction = {
    "transaction_id": "T123",
    "customer_id": "CUST1092",
    "transaction_amount": 200,
    "transaction_time": "2025-09-04 12:30:00",
    "device_type": "Mobile",
    "location": "Lagos",
    "transaction_type": "Transfer",
    "is_foreign_transaction": 0,
    "previous_fraud_flag": 0,
    "is_high_risk_country": 0,
    "day_of_week": "Fri",
    "time_of_day": "Afternoon",
    "risk_score": 2.5
}

# Send POST request to the API
response = requests.post(url, json=transaction)

# Parse and print results
if response.ok:
    result = response.json()
    print("Prediction:", result["prediction"])            # 0 = Not Fraud, 1 = Fraud
    print("Fraud Probability:", result["fraud_probability"])
else:
    print("Error:", response.status_code, response.text)
```

### Run the Streamlit App

Launch the interactive fraud dashboard:

```bash
streamlit run streamlit_app.py
```
Or use the deployed online web app: [Fraud-Detector](https://fintech-fraud-ai-ozechi-0.streamlit.app/)

---

## Assumptions and Limitations

* Dataset (1,000 samples) may not reflect real-world fraud complexity.
* Temporal leakage risks are mitigated with proper splits, but real-time drift must be monitored.
* Model assumes features provided at transaction time — some external signals (e.g., IP data, network graph features) not included.

---

## Future Improvements

* Explore **Autoencoder anomaly detection** for temporal fraud sequences.
* Integrate **active learning feedback loop** from fraud analysts.
* Expand dataset and evaluate with **real-world transactions**.

---
