"""Model training and persistence helpers."""

import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def train_model(X, y):
    """Train and return a scikit-learn pipeline.

    Inputs:
      - X: DataFrame or array-like of features
      - y: array-like of labels (binary)

    Returns: trained Pipeline
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipeline.fit(X, y)
    return pipeline


def save_model(model, path: str):
    """Save trained model to `path` using joblib."""
    joblib.dump(model, path)


def load_model(path: str):
    """Load a model saved with `save_model`."""
    return joblib.load(path)
