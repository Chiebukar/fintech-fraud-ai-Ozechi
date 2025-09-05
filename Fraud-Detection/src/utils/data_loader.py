import pandas as pd
from pathlib import Path
from typing import Optional

def load_data(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if p.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(p)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(p, sheet_name=sheet_name)
        except Exception as e:
            raise RuntimeError("Install Excel readers: pip install xlrd openpyxl") from e
    raise ValueError(f"Unsupported file type: {p.suffix}")
