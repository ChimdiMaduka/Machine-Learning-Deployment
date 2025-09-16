US Visa Application Outcome Classifier

Lightweight Streamlit scaffold for predicting US visa application outcomes (Certified or Denied).

Quick start (local)

1. Create and activate a Python virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run streamlit_app.py
```

Training

Train a model using the included sample data and save it to `models/model.joblib`:

```powershell
python -m src.train --data data/sample.csv --out models/model.joblib
```

Testing

Run the test suite with pytest (inside the activated venv):

```powershell
pytest -q
```

Docker

Build and run the image locally:

```powershell
docker build -t visa-classifier:local .
docker run -p 8501:8501 visa-classifier:local
```

Deploying to Streamlit Cloud or other PaaS

- Streamlit Cloud: connect the GitHub repo and set the start command to `streamlit run streamlit_app.py`.
- Heroku-like platforms: `Procfile` is included.

Responsible ML notes

- Do not commit real applicant data or PII. Always store sensitive data securely outside the repository and use environment variables for credentials.

Project structure

See the repository root for `src/`, `data/`, `models/`, `notebooks/`, and `tests/`.
