"""
Feature engineering transformers using sklearn API so they can be integrated into pipelines.

Each transformer handles both historical and new-customer cases:
- If there is no history for a customer, safe defaults are returned:
  - past_mean_amount → transaction amount itself
  - recency_ratio → 1.0
  - switch risks → "unchanged"
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


# ------------------------
# Column selector
# ------------------------
class ColumnSelector(TransformerMixin, BaseEstimator):
    """
    Selects specific columns from a dataframe.

    Parameters
    ----------
    columns : list of str, optional
        The columns to keep. If None, all columns are returned.
    """
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.columns is None:
            return X
        return X[self.columns]

# ------------------------
# Transaction amount filter
# ------------------------
class TransactionAmountFilter(BaseEstimator, TransformerMixin):
    """Remove rows where transaction_amount < 0."""

    def __init__(self, amount_col="transaction_amount"):
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Defensive copy
        df = X.copy()
        # Filter out negatives
        df = df[df[self.amount_col] >= 0].reset_index(drop=True)
        return df

# ------------------------
# Datetime feature extraction
# ------------------------
class DatetimeTransformer(TransformerMixin, BaseEstimator):
    """
    Extracts useful datetime features from a timestamp column.

    Features added:
    - _hour : hour of day
    - _month : month number
    - _is_weekend : 1 if Sat/Sun else 0

    Parameters
    ----------
    datetime_column : str
        Name of the datetime column.
    """
    def __init__(self, datetime_column: str):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_column] = pd.to_datetime(X[self.datetime_column], errors="coerce")
        X["_hour"] = X[self.datetime_column].dt.hour.fillna(-1).astype(int)
        X["_month"] = X[self.datetime_column].dt.month.fillna(-1).astype(int)
        X["_is_weekend"] = (X[self.datetime_column].dt.dayofweek >= 5).fillna(-1).astype(int)
        return X


# ------------------------
# NaN filler
# ------------------------
class NaNFiller(TransformerMixin, BaseEstimator):
    """
    Fills missing values in a dataframe.

    Parameters
    ----------
    fill_value : float, default=0
        Value to replace NaNs with.
    """
    def __init__(self, fill_value: float = 0):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.fillna(self.fill_value)


# ------------------------
# Log-transform transaction amount
# ------------------------
class LogAmountTransformer(TransformerMixin, BaseEstimator):
    """
    Adds log-transformed transaction amount.

    Feature added:
    - log_<amount_col>

    Parameters
    ----------
    amount_col : str, default="transaction_amount"
        Column containing transaction amounts.
    """
    def __init__(self, amount_col: str = "transaction_amount"):
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["log_" + self.amount_col] = np.log1p(X[self.amount_col])
        return X


# ------------------------
# Recency ratio
# ------------------------
class RecencyRatioTransformer(TransformerMixin, BaseEstimator):
    """
    Computes recency ratio: rolling avg of recent transaction time differences
    divided by avg nonfraud time differences.

    Parameters
    ----------
    customer_col : str, default="customer_id"
    time_col : str, default="transaction_time"
    label_col : str, default="label_code"

    Notes
    -----
    - Sorts by (customer_id, transaction_time).
    - For new customers (no history), defaults to 1.0.
    """
    def __init__(self, customer_col="customer_id", time_col="transaction_time", label_col="label_code"):
        self.customer_col = customer_col
        self.time_col = time_col
        self.label_col = label_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
        df = df.sort_values(by=[self.customer_col, self.time_col]).copy()

        df["time_diff_min"] = (
            df.groupby(self.customer_col)[self.time_col]
              .diff()
              .dt.total_seconds() / 60
        )

        df["avg_time_nonfraud"] = (
            df.assign(nonfraud_time_diff=df.where(df[self.label_col] == 0)["time_diff_min"])
              .groupby(self.customer_col)["nonfraud_time_diff"]
              .expanding()
              .mean()
              .reset_index(level=0, drop=True)
        )

        df["rolling3_time"] = (
            df.groupby(self.customer_col)["time_diff_min"]
              .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )

        df["recency_ratio"] = df["rolling3_time"] / df["avg_time_nonfraud"]
        df["recency_ratio"] = df["recency_ratio"].replace([np.inf, -np.inf], 1.0).fillna(1.0)

        return df


# ------------------------
# Amount relative to past mean
# ------------------------
class AmountRelativeToHistoryTransformer(TransformerMixin, BaseEstimator):
    """
    Computes past mean of transaction amounts for each customer.

    Feature added:
    - past_mean_amount

    Parameters
    ----------
    customer_col : str, default="customer_id"
    time_col : str, default="transaction_time"
    amount_col : str, default="transaction_amount"
    label_col : str, default="label_code"

    Notes
    -----
    - Sorts by (customer_id, transaction_time).
    - Uses only nonfraud transactions for history.
    - For new customers → defaults to current transaction amount.
    """
    def __init__(self, customer_col="customer_id", time_col="transaction_time",
                 amount_col="transaction_amount", label_col="label_code"):
        self.customer_col = customer_col
        self.time_col = time_col
        self.amount_col = amount_col
        self.label_col = label_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df_sorted = df.sort_values([self.customer_col, self.time_col])

        def per_user(group):
            mask = group[self.label_col] == 0
            if mask.sum() == 0:  # new customer
                return pd.Series(group[self.amount_col], index=group.index)

            past_mean = group.loc[mask, self.amount_col].expanding().mean().shift()
            past_mean = past_mean.reindex(group.index)
            return past_mean.fillna(group[self.amount_col])

        past_mean_series = (
            df_sorted.groupby(self.customer_col, group_keys=False).apply(per_user)
        )

        # Ensure past_mean_series is 1D
        if isinstance(past_mean_series, pd.DataFrame):
            past_mean_series = past_mean_series.iloc[:, 0]

        df["past_mean_amount"] = past_mean_series.reindex(df.index)
        return df


# ------------------------
# Switch risk (geo/device/type)
# ------------------------
class SwitchRiskTransformer(TransformerMixin, BaseEstimator):
    """
    Computes risk level of switching categorical features (e.g., device, location, transaction type).

    Adds a new column:
        <category_col>_switch_risk

    Categories:
        - unchanged
        - very_quick_change (<=2 min)
        - quick_change (<=5 min)
        - medium_change (<=60 min)
        - reasonable_change (<=360 min)
        - very_reasonable_change (>360 min)

    Parameters
    ----------
    customer_col : str, default="customer_id"
        Column identifying unique customers.
    time_col : str, default="transaction_time"
        Column containing datetime of transaction.
    category_col : str, default="location"
        Column to track for switching.
    label_col : str, default="label_code"
        Column identifying fraudulent transactions (0 = non-fraud).
    out_col : str, optional
        Custom name for output column. Defaults to <category_col>_switch_risk.

    Notes
    -----
    - Sorts by (customer_id, transaction_time).
    - Uses last non-fraud transaction as baseline.
    - For new customers or missing values → defaults to "unchanged".
    """
    def __init__(self, customer_col="customer_id", time_col="transaction_time",
                 category_col="location", label_col="label_code", out_col=None):
        self.customer_col = customer_col
        self.time_col = time_col
        self.category_col = category_col
        self.label_col = label_col
        self.out_col = out_col or f"{category_col}_switch_risk"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        out = pd.Series("unchanged", index=df.index, dtype="object")

        if df.empty:
            df[self.out_col] = out
            return df

        # Ensure datetime
        df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
        # Normalize categorical column
        df["_norm_cat"] = df[self.category_col].astype(str).str.strip().str.lower().replace("nan", "")

        def process_group(g: pd.DataFrame) -> pd.Series:
            """Compute switch risk per customer group"""
            g = g.sort_values(self.time_col)
            nonfraud_mask = g[self.label_col] == 0

            # If no non-fraud history → all 'unchanged'
            if nonfraud_mask.sum() == 0:
                return pd.Series(["unchanged"] * len(g), index=g.index)

            # Last non-fraud transaction
            prev_cat = g["_norm_cat"].where(nonfraud_mask).ffill().shift(1)
            prev_time = g[self.time_col].where(nonfraud_mask).ffill().shift(1)

            loc_changed = prev_cat.notna() & (g["_norm_cat"] != prev_cat)
            time_diff_min = (g[self.time_col] - prev_time).dt.total_seconds() / 60.0

            def categorize(change: bool, diff: float) -> str:
                if not change or pd.isna(diff):
                    return "unchanged"
                if diff <= 2:
                    return "very_quick_change"
                elif diff <= 5:
                    return "quick_change"
                elif diff <= 60:
                    return "medium_change"
                elif diff <= 360:
                    return "reasonable_change"
                else:
                    return "very_reasonable_change"

            return pd.Series([categorize(c, d) for c, d in zip(loc_changed, time_diff_min)], index=g.index)
            
        # Apply per customer
        result = df.groupby(self.customer_col, group_keys=False).apply(process_group)

        # Align indices with original df
        result = result.reindex(df.index)

        # Assign to output column
        out.loc[:] = result.fillna("unchanged")
        df[self.out_col] = out
        df.drop(columns="_norm_cat", inplace=True)
        return df
