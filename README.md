# RetailNova Serverless FinOps Dashboard

A Streamlit dashboard for analyzing serverless (Lambda/Functions/Cloud Functions) costs.

## Quick Start (Local)

1. Create a venv and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run ServerlessComputing.py
```

## Deploying to Streamlit Cloud

1. Ensure `requirements.txt` is present in the root of the repository (it is in this repo).
2. Push your changes to GitHub.
3. In Streamlit Cloud, create a new app and point to the `ServerlessComputing.py` file.

If you encounter `ModuleNotFoundError` errors in the deployed app, ensure the missing package is listed in `requirements.txt` and then re-deploy.

## Files

- `ServerlessComputing.py` - main Streamlit app
- `Serverless_Data.csv` - sample data (used by the app)
- `requirements.txt` - Python dependencies for deployment

## Notes

- If you see errors for `plotly` or other libraries on deployment, add them to `requirements.txt` and re-deploy.
