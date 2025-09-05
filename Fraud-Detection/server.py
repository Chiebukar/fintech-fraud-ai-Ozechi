"""
server.py

Flask server exposing a REST API for fraud scoring.

- Accepts JSON payload with transaction data
- Returns predicted fraud probability and label
"""

from flask import Flask, request, jsonify
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / "src" / "utils")) # Add src/utils to Python path
from score import score # Now you can import score

# ------------------------
# Config
# ------------------------
PIPELINE_PATH = Path("./src/models/preprocess_and_model.pkl")
HISTORICAL_DATA_PATH = Path("./src/data/fintech_sample_fintech_transactions.xlsx")

app = Flask(__name__)

# ------------------------
# Routes
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects a JSON payload representing a single transaction.
    Example payload:
    {
        "transaction_id": "T123",
        "customer_id": "CUST0000",
        "transaction_amount": 20,
        "transaction_time": "2025-09-04 12:30:00",
        "device_type": "Mobile",
        "location": "Lagos",
        "transaction_type": "Transfer",
        "is_foreign_transaction": 1,
        "previous_fraud_flag": 0,
        "is_high_risk_country": 1,
        "day_of_week": "Fri",
        "time_of_day": "Afternoon",
        "risk_score": 9.5
    }
    """
    try:
        # Parse JSON payload
        tx_json = request.get_json()
        if not tx_json:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Convert to DataFrame
        tx_df = pd.DataFrame([tx_json])

        # Score transaction
        pred, prob = score(
            new_tx=tx_df,
            pipeline_path=PIPELINE_PATH,
            historical_data_path=HISTORICAL_DATA_PATH
        )

        # Return JSON response
        return jsonify({
            "prediction": int(pred),
            "fraud_probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------
# Run server
# ------------------------
if __name__ == "__main__":
    # Flask default port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
