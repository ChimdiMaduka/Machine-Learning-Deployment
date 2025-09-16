import os
from src.train import train_and_save


def test_train_and_save_creates_model(tmp_path):
    data = os.path.join(os.getcwd(), "data", "sample.csv")
    out = tmp_path / "model.joblib"
    model_path = train_and_save(data, str(out))
    assert os.path.exists(model_path)
