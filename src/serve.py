"""Small serving helpers for predictions."""
import os
import pandas as pd
from .model import load_model


def predict_from_df(df: pd.DataFrame, model_path: str = "models/model.joblib"):
    if os.path.exists(model_path):
        model = load_model(model_path)
        preds = model.predict(df)
        return preds
    # fallback rule
    if "PREVAILING_WAGE" in df.columns:
        return [("Certified" if v >= 50000 else "Denied") for v in df["PREVAILING_WAGE"]]
    return ["Certified"] * len(df)


def predict_from_csv(path: str, model_path: str = "models/model.joblib"):
    df = pd.read_csv(path)
    return predict_from_df(df, model_path)
