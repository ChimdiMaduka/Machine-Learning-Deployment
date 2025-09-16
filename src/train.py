"""Training script for the visa classifier.

Provides a programmatic `train_and_save` function and CLI entrypoint.
"""
from pathlib import Path
import argparse
import logging

from .preprocessing import load_data, preprocess
from .model import train_model, save_model


def train_and_save(data_path: str, model_path: str):
    logging.info(f"Loading data from {data_path}")
    df = load_data(data_path)
    X, y = preprocess(df)
    if y is None:
        raise ValueError("Training requires a labeled dataset with `CASE_STATUS` column")
    model = train_model(X, y)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, model_path)
    return model_path


def _cli():
    p = argparse.ArgumentParser(description="Train visa classifier and save model")
    p.add_argument("--data", default="data/sample.csv")
    p.add_argument("--out", default="models/model.joblib")
    args = p.parse_args()
    train_and_save(args.data, args.out)


if __name__ == "__main__":
    _cli()
