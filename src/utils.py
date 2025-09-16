"""Small utilities used across the project."""

import os


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def models_dir():
    return os.path.join(os.getcwd(), "models")
