# train.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_sample_weight
from utils import extract_and_save_distinct_values

print("Importing custom transformers...")
# Import custom transformers
from features import (
    TransactionAmountFilter,
    NaNFiller,
    LogAmountTransformer,
    RecencyRatioTransformer,
    AmountRelativeToHistoryTransformer,
    SwitchRiskTransformer,
)

# ----------------------------
# Config paths
# ----------------------------
DATA_PATH = "./src/data/fintech_sample_fintech_transactions.xlsx"
PIPELINE_PATH = "./src/models/preprocess_and_model.pkl"

# ----------------------------
# Load data
# ----------------------------
print("Loading historical dataset...")
df = pd.read_excel(DATA_PATH, engine="openpyxl")

# ----------------------------
# Define features used
# ----------------------------
numeric_features = [
    "risk_score",
    "log_transaction_amount",
    "past_mean_amount",
    "recency_ratio",
]
categorical_features = [
    "transaction_type_switch",
    "device_switch_risk",
    "geo_switch_risk",
]
binary_features = [
    "is_foreign_transaction",
    "previous_fraud_flag",
    "is_high_risk_country",
]
target_col = "label_code"

# ----------------------------
# Build feature engineering pipeline
# ----------------------------
print("Building feature engineering pipeline...")
feature_steps = Pipeline(steps=[
    ("filter_amount", TransactionAmountFilter(amount_col="transaction_amount")),
    ("nan_filler", NaNFiller()),
    ("log_amount", LogAmountTransformer(amount_col="transaction_amount")),
    ("recency_ratio", RecencyRatioTransformer(
        customer_col="customer_id", time_col="transaction_time", label_col=target_col
    )),
    ("past_mean", AmountRelativeToHistoryTransformer(
        customer_col="customer_id", time_col="transaction_time",
        amount_col="transaction_amount", label_col=target_col
    )),
    ("geo_switch", SwitchRiskTransformer(
        customer_col="customer_id", time_col="transaction_time",
        category_col="location", label_col=target_col, out_col="geo_switch_risk"
    )),
    ("device_switch", SwitchRiskTransformer(
        customer_col="customer_id", time_col="transaction_time",
        category_col="device_type", label_col=target_col, out_col="device_switch_risk"
    )),
    ("txn_type_switch", SwitchRiskTransformer(
        customer_col="customer_id", time_col="transaction_time",
        category_col="transaction_type", label_col=target_col, out_col="transaction_type_switch"
    )),
])

# Preprocessing for model
print("Building preprocessing pipeline...")
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (("cat", OrdinalEncoder(), categorical_features)),
        ("bin", "passthrough", binary_features),
    ]
)

# ----------------------------
# Apply feature engineering and preprocessing
# ----------------------------
print("Applying feature engineering...")
df_transformed = feature_steps.fit_transform(df)

# Apply preprocessing to convert to numeric
print("Applying preprocessing...")
X = pd.DataFrame(
    preprocess.fit_transform(df_transformed),
    columns=preprocess.get_feature_names_out()
)

# Target
y = df_transformed[target_col]


# ----------------------------
# Compute instance weights
# ----------------------------
print("Computing sample weights...")
# Base class weights
sample_weights = compute_sample_weight("balanced", y)

# Initial model to estimate disagreements
init_model = GradientBoostingClassifier(random_state=42)
init_model.fit(X, y, sample_weight=sample_weights)
p_train = init_model.predict_proba(X)[:, 1]

disagree = np.abs(p_train - y)
noise_downweight = 1.0 / (1.0 + 5 * disagree)
final_weights = sample_weights * noise_downweight

# ----------------------------
# Train final model with early stopping
# ----------------------------
print("Training Gradient Boosting model...")

# Time-based split (80/20)
split_idx = int(0.8 * len(X))
X_tr, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_tr, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
w_tr, w_val = final_weights[:split_idx], final_weights[split_idx:]

gb_model = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=50,
    subsample=0.8,
    random_state=42,
)

gb_model.fit(X_tr, y_tr, sample_weight=w_tr)

best_iter, best_loss = 0, np.inf
for i, y_pred in enumerate(gb_model.staged_predict_proba(X_val)):
    loss = log_loss(y_val, y_pred[:, 1], sample_weight=w_val)
    if loss < best_loss:
        best_loss, best_iter = loss, i

print(f"Best iteration (early stop) = {best_iter}, best log loss = {best_loss:.4f}")

# Refit with best iteration
gb_model.set_params(n_estimators=best_iter + 1)
gb_model.fit(X_tr, y_tr, sample_weight=w_tr)

# ----------------------------
# Full pipeline = features + preprocess + model
# ----------------------------
pipeline = Pipeline(steps=[
    ("features", feature_steps),
    ("preprocess", preprocess),
    ("model", gb_model),
])

# ----------------------------
# Save pipeline
# ----------------------------
print(f"Saving full pipeline to {PIPELINE_PATH}")
joblib.dump(pipeline, PIPELINE_PATH)


# ----------------------------
# Save Feature Values
# ----------------------------
feature_values = extract_and_save_distinct_values()
print(f"Feature values saved")


if __name__ == "__main__":
    print("Training complete.")
