# streamlit_app.py
"""
Streamlit frontend for fraud scoring.

Usage:
    streamlit run streamlit_app.py

Assumes:
- ./src/data/distinct_feature_values.json exists (produced by your util)
- score.score(...) function available at src/utils/score.py
- The score(...) function takes a pandas DataFrame containing a single transaction
  and returns (prediction:int, probability:float)
"""

import sys
from pathlib import Path
import json
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

# ---- make src/utils importable ----
ROOT = Path(__file__).parent.resolve()
SYS_PATH_INSERT = str(ROOT / "src" / "utils")
if SYS_PATH_INSERT not in sys.path:
    sys.path.insert(0, SYS_PATH_INSERT)

# Import score function (from src/utils/score.py)
try:
    from score import score as score_transaction  # score(new_tx: pd.DataFrame) -> (pred, prob)
except Exception as e:
    st.error(f"Could not import score.score(). Make sure src/utils is on PYTHONPATH. Error: {e}")
    raise

# ---- constants / defaults ----
DISTINCT_JSON = ROOT / "src" / "features" / "feature_values.json"
DEFAULT_LOOKAHEAD_DAYS = 30  # random datetime up to N days in future

TIME_OF_DAY_LABELS = [
    "Morning",   # 05:00 - 11:59
    "Afternoon", # 12:00 - 16:59
    "Evening",   # 17:00 - 20:59
    "Night"      # 21:00 - 04:59
]


@st.cache_data(show_spinner=False)
def load_distinct_values(path: Path = DISTINCT_JSON):
    if not path.exists():
        st.warning(f"Distinct features file not found at {path}. Dropdowns will be empty.")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def random_future_datetime(start: datetime = None, max_days: int = DEFAULT_LOOKAHEAD_DAYS):
    start = start or datetime.now()
    delta_days = random.randint(0, max_days)
    # random hour/minute/second within day
    rand_seconds = random.randint(0, 86399)
    dt = start + timedelta(days=delta_days, seconds=rand_seconds)
    return dt.replace(microsecond=0)


def time_of_day_from_dt(dt: datetime) -> str:
    h = dt.hour
    if 5 <= h <= 11:
        return "Morning"
    if 12 <= h <= 16:
        return "Afternoon"
    if 17 <= h <= 20:
        return "Evening"
    return "Night"


def day_of_week_from_dt(dt: datetime) -> str:
    # Return abbreviated day like "Mon", "Tue", "Wed" or "Fri" as in sample
    return dt.strftime("%a")  # e.g., 'Fri'


def build_transaction_dict(
    transaction_id: str,
    customer_id: str,
    transaction_amount: float,
    transaction_time: datetime,
    device_type: str,
    location: str,
    transaction_type: str,
    is_foreign_transaction: int,
    previous_fraud_flag: int,
    is_high_risk_country: int,
    risk_score: float = 0.0,
):
    tx_time_str = transaction_time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "transaction_amount": float(transaction_amount),
        "transaction_time": tx_time_str,
        "device_type": device_type,
        "location": location,
        "transaction_type": transaction_type,
        "is_foreign_transaction": int(is_foreign_transaction),
        "previous_fraud_flag": int(previous_fraud_flag),
        "is_high_risk_country": int(is_high_risk_country),
        "day_of_week": day_of_week_from_dt(transaction_time),
        "time_of_day": time_of_day_from_dt(transaction_time),
        "risk_score": float(risk_score),
    }


# ---- Streamlit UI ----
st.set_page_config(page_title="Fraud Scorer", page_icon="🛡️", layout="wide")
st.title("🛡️ Fraud Detection — Live Scoring")
st.markdown(
    """
    Use the controls on the left to craft a transaction. Press **Score Transaction** to get
    the fraud probability and predicted label from the trained model.
    """
)

distinct_vals = load_distinct_values()

# Sidebar inputs
with st.sidebar:
    st.header("Transaction input")

    # customer_id: searchable dropdown (selectbox supports filtering)
    cust_options = distinct_vals.get("customer_id", [])
    customer_id = st.selectbox(
        "Customer ID",
        options=["(new)"] + cust_options,
        index=0,
        help="Select an existing customer or '(new)' to enter a new customer id."
    )
    if customer_id == "(new)":
        customer_id = st.text_input("Enter new Customer ID", value=f"CUST{random.randint(1000,9999)}")

    # transaction_type, device_type, location (searchable dropdowns)
    txn_type_opts = distinct_vals.get("transaction_type", [])
    transaction_type = st.selectbox("Transaction type", options=txn_type_opts, index=0 if txn_type_opts else -1)

    device_opts = distinct_vals.get("device_type", [])
    device_type = st.selectbox("Device type", options=device_opts, index=0 if device_opts else -1)

    location_opts = distinct_vals.get("location", [])
    location = st.selectbox("Location", options=location_opts, index=0 if location_opts else -1)

    st.markdown("---")
    # Random future datetime selector
    st.write("Transaction date/time")
    rand_dt = random_future_datetime()
    # Let user choose to use random or custom
    use_random_dt = st.checkbox("Choose random future date/time", value=True)
    if use_random_dt:
        # show & allow tweak
        dt = st.date_input("Date", value=rand_dt.date())
        t = st.time_input("Time", value=rand_dt.time())
        # combine
        transaction_time = datetime.combine(dt, t)
    else:
        dt = st.date_input("Date", value=rand_dt.date())
        t = st.time_input("Time", value=rand_dt.time())
        transaction_time = datetime.combine(dt, t)

    st.markdown("---")
    transaction_amount = st.number_input("Transaction amount", min_value=0.0, value=50.0, step=1.0, format="%.2f")
    is_foreign = st.selectbox("Foreign transaction?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    prev_fraud = st.selectbox("Previous fraud flag", options=[0, 1], format_func=lambda x: "Not Fraud" if x == 0 else "Fraud")
    is_high_risk_country = st.selectbox("High risk country?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    risk_score = st.number_input("Risk score (optional)", value=0.0, format="%.3f")
    st.markdown("---")
    if st.button("Score Transaction"):
        st.session_state["score_now"] = True
    if st.button("Randomize Date/Time"):
        st.session_state["randomize_dt"] = True

# react to randomize button
if st.session_state.get("randomize_dt", False):
    transaction_time = random_future_datetime()
    st.session_state["randomize_dt"] = False

# Ensure we have transaction_time variable
try:
    transaction_time  # noqa
except NameError:
    transaction_time = random_future_datetime()

# Main panel: show transaction summary and scoring result
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Transaction preview")
    # Generate a transaction id if missing
    tx_id = f"T{random.randint(100000,999999)}"
    preview_tx = build_transaction_dict(
        transaction_id=tx_id,
        customer_id=customer_id,
        transaction_amount=transaction_amount,
        transaction_time=transaction_time,
        device_type=device_type or "",
        location=location or "",
        transaction_type=transaction_type or "",
        is_foreign_transaction=is_foreign,
        previous_fraud_flag=prev_fraud,
        is_high_risk_country=is_high_risk_country,
        risk_score=risk_score,
    )

    st.markdown("**Preview (what will be scored)**")
    st.json(preview_tx)

with col2:
    st.subheader("Quick info")
    st.metric("Time of transaction", preview_tx["transaction_time"])
    st.metric("Day of week", preview_tx["day_of_week"])
    st.metric("Time of day", preview_tx["time_of_day"])

# Score when requested
if st.session_state.get("score_now", False):
    st.session_state["score_now"] = False

    # Build dataframe expected by score.score()
    new_tx_df = pd.DataFrame([preview_tx])

    st.info("Scoring... this may take a moment (pipeline & model load).")

    try:
        pred, prob = score_transaction(new_tx_df)  # uses default history and pipeline path inside your score.py
    except Exception as e:
        st.error(f"Scoring failed: {e}")
    else:
        # show results with nice visuals
        st.success("Scoring complete")
        prob_pct = float(prob) * 100.0
        pred = int(pred)

        rcol1, rcol2 = st.columns([1, 2])
        with rcol1:
            label_text = "FRAUD" if pred == 1 else "NOT FRAUD"
            label_color = "🔴" if pred == 1 else "🟢"
            st.markdown(f"### {label_color} {label_text}")
            st.metric("Fraud probability", f"{prob_pct:.2f} %")
        with rcol2:
            st.markdown("#### Probability gauge")
            # Simple horizontal bar using st.progress (0-1)
            st.progress(min(max(float(prob), 0.0), 1.0))

        # More details
        with st.expander("Show full model output / details"):
            st.write("Prediction label:", pred)
            st.write("Fraud probability (float 0-1):", float(prob))
            st.json(preview_tx)

    # optional: append to session history
    history = st.session_state.get("transactions_scored", [])
    history.append({"tx": preview_tx, "prediction": pred, "probability": float(prob)})
    st.session_state["transactions_scored"] = history

# show recent scored transactions in a table
if st.session_state.get("transactions_scored"):
    st.markdown("---")
    st.subheader("Recent scored transactions (session)")
    recent = pd.DataFrame([
        {
            "transaction_id": h["tx"]["transaction_id"],
            "customer_id": h["tx"]["customer_id"],
            "amount": h["tx"]["transaction_amount"],
            "time": h["tx"]["transaction_time"],
            "pred": h["prediction"],
            "prob": h["probability"],
        }
        for h in st.session_state["transactions_scored"]
    ])
    st.dataframe(recent)

# footer
st.markdown("---")
st.caption("Streamlit UI • Fraud detection demo • powered by your trained pipeline")
