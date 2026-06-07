# Fraud Detection

End-to-end fraud detection pipeline covering data preprocessing, EDA, feature engineering, modeling, and explainability.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

| Path | Description |
|------|-------------|
| `data/raw/` | Original datasets (gitignored) |
| `data/processed/` | Cleaned and feature-engineered data |
| `notebooks/` | EDA, feature engineering, modeling, SHAP notebooks |
| `src/` | Reusable source modules |
| `scripts/` | Standalone execution scripts |
| `tests/` | Unit tests |
| `models/` | Saved model artifacts |

## Datasets

- `Fraud_Data.csv` — e-commerce transaction data with fraud labels
- `creditcard.csv` — bank transaction data (PCA-transformed features)
- `IpAddress_to_Country.csv` — IP range to country mapping

## Notebooks

1. `eda-fraud-data.ipynb` — EDA on Fraud_Data.csv
2. `eda-creditcard.ipynb` — EDA on creditcard.csv
3. `feature-engineering.ipynb` — Feature engineering and preprocessing
4. `modeling.ipynb` — Model training and evaluation
5. `shap-explainability.ipynb` — SHAP-based model explainability
