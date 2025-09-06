import pandas as pd
import json
from pathlib import Path
from typing import List, Union, Dict


# ----------------------------
# Config paths
# ----------------------------
DATA_PATH = "./src/data/fintech_sample_fintech_transactions.xlsx"
OUTPUT_PATH = "./src/features/feature_values.json"
REQUIRED_FEATURES = [
    "customer_id",
    "transaction_type",
    "device_type",
    "location",
]


def extract_and_save_distinct_values(
    dataset_path: Union[str, Path] = DATA_PATH,
    required_features: List[str] = REQUIRED_FEATURES,
    output_path: Union[str, Path] = OUTPUT_PATH,
) -> Dict[str, Union[List, Dict]]:
    """
    Extract distinct values for given features in a dataset and save them to JSON.

    Args:
        dataset_path (str | Path): Path to the dataset (Excel or CSV).
        required_features (List[str]): List of features to extract distinct values for.
                                       Defaults to customer_id, transaction_type,
                                       device_type, location, transaction_time.
        output_path (str | Path): Path to save JSON file of distinct values.

    Returns:
        dict: Dictionary of distinct values for each feature.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    # Default required features
    if required_features is None:
        required_features = [
            "customer_id",
            "transaction_type",
            "device_type",
            "location",
            "transaction_time",
        ]

    # Load dataset 
    df = pd.read_excel(dataset_path, engine="openpyxl")
    

    distinct_values = {}
    for col in required_features:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataset.")

        # Extract distinct values
        distinct_values[col] = sorted(df[col].dropna().unique().tolist())

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(distinct_values, f, indent=4)

    return distinct_values