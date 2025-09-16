import os
import pandas as pd
from src.serve import predict_from_csv


def test_predict_from_csv_returns_list():
    data = os.path.join(os.getcwd(), "data", "sample.csv")
    preds = predict_from_csv(data)
    assert isinstance(preds, (list,))
    assert len(preds) > 0
