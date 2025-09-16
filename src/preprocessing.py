"""Preprocessing utilities for the Visa classifier.

This module keeps preprocessing logic small and testable. Replace with
project-specific feature engineering as needed.
"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV from path into a DataFrame."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame):
    """Return features X and optional labels y.

    Current simple implementation:
    - Uses `PREVAILING_WAGE` as the numeric feature (fills missing with 0)
    - If `CASE_STATUS` exists, returns y as binary (1=CERTIFIED, 0=other)
    """
    df = df.copy()
    if "PREVAILING_WAGE" in df.columns:
        X = df[["PREVAILING_WAGE"]].fillna(0)
    else:
        X = pd.DataFrame({"PREVAILING_WAGE": [0] * len(df)})

    y = None
    if "CASE_STATUS" in df.columns:
        y = (df["CASE_STATUS"].str.upper() == "CERTIFIED").astype(int)

    return X, y
