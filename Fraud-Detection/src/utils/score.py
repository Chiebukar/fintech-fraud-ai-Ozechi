"""
score.py

Robust scoring for a new transaction using the trained fraud detection pipeline.

- Handles multi-column outputs from custom transformers.
- Works even if the customer has no history.
- Avoids NaN issues.
- Ensures the prediction corresponds exactly to the new transaction.
- Updates historical dataset on disk.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ------------------------
# Config paths
# ------------------------
PIPELINE_PATH = Path("./src/models/preprocess_and_model.pkl")
HISTORICAL_DATA_PATH = Path("./src/data/fintech_sample_fintech_transactions.xlsx")


def score(new_tx: pd.DataFrame,
          pipeline_path: str = PIPELINE_PATH,
          historical_data_path: str = HISTORICAL_DATA_PATH,
          label_col: str = "label_code",
          customer_col: str = "customer_id") -> tuple:
    """
    Score a single new transaction using the trained fraud detection pipeline.
    """
    # ------------------------
    # Validate input
    # ------------------------
    if not isinstance(new_tx, pd.DataFrame):
        raise ValueError("new_tx must be a pandas DataFrame.")
    if len(new_tx) != 1:
        raise ValueError("new_tx must contain exactly one row.")

    # ------------------------
    # Load historical data
    # ------------------------
    historical_data = pd.read_excel(historical_data_path, engine="openpyxl")

    # ------------------------
    # Add temporary ID to track new transaction
    # ------------------------
    new_tx = new_tx.copy()
    new_tx["_temp_id"] = "NEW_TX"
    new_tx[label_col] = 0  # dummy label

    # Combine historical data + new transaction
    batch = pd.concat([historical_data, new_tx], ignore_index=True)

    # ------------------------
    # Load pipeline
    # ------------------------
    pipeline = joblib.load(pipeline_path)

    # ------------------------
    # Feature engineering
    # ------------------------
    try:
        batch_transformed = pipeline.named_steps['features'].transform(batch)
    except Exception as e:
        raise RuntimeError(f"Error in feature engineering: {e}")

    # ------------------------
    # Extract new transaction from transformed batch
    # ------------------------
    new_tx_transformed = batch_transformed[batch_transformed["_temp_id"] == "NEW_TX"].drop(columns=["_temp_id"])

    # ------------------------
    # Preprocessing + Model
    # ------------------------
    try:
        Xt = pipeline.named_steps['preprocess'].transform(new_tx_transformed)
        preds_proba = pipeline.named_steps['model'].predict_proba(Xt)
        preds_label = pipeline.named_steps['model'].predict(Xt)
    except Exception as e:
        raise RuntimeError(f"Error during pipeline prediction: {e}")

    # ------------------------
    # Extract predictions
    # ------------------------
    prediction = int(preds_label[0])
    probability = float(preds_proba[0, 1])  # probability of fraud

    # ------------------------
    # Update historical dataset
    # ------------------------
    new_tx_scored = new_tx.copy()
    new_tx_scored[label_col] = prediction
    new_tx_scored = new_tx_scored.drop(columns=["_temp_id"])

    # updated_history = pd.concat([historical_data, new_tx_scored], ignore_index=True)
    # updated_history.to_excel(historical_data_path, index=False, engine="openpyxl")

    return prediction, round(probability, 2)


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    new_tx = pd.DataFrame([{
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
    }])

    print("Scoring new transaction...")
    pred, prob = score(new_tx=new_tx)
    print(f"Prediction (0=not fraud, 1=fraud): {pred}")
    print(f"Fraud probability: {prob:.2f}")
