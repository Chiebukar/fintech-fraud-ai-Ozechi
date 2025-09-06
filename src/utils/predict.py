# predict_cli.py
"""
Command-line tool for scoring transactions from an Excel file

Usage:
    python predict.py --input ./path/to/new_sample.xlsx
"""

import argparse
import pandas as pd
from pathlib import Path

# Import score function from score.py
from score import score


def main():
    parser = argparse.ArgumentParser(description="Score transactions from an Excel file.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the Excel file containing the transaction(s) to score."
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load the input Excel file
    df = pd.read_excel(input_path, engine="openpyxl")

    if len(df) == 0:
        raise ValueError("Input file is empty.")
    if len(df) > 1:
        print(f"⚠️ Warning: Found {len(df)} rows. Only the first will be scored.")
        df = df.iloc[[0]]  # Keep only the first row

    # Call score function
    pred, prob = score(new_tx=df)

    # Output results
    print("\n--- Fraud Scoring Result ---")
    print(f"Prediction: {'FRAUD' if pred == 1 else 'NOT FRAUD'}")
    print(f"Fraud Probability: {prob:.2f}\n")


if __name__ == "__main__":
    main()
