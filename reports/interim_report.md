# Interim Report — End-to-End Fraud Detection Pipeline

**Author:** Ephrata Yohannes  
**Project:** Fraud Detection — Machine Learning Pipeline  
**Repository:** https://github.com/EphrataTech/Fraud_detection  
**Date:** June 2025

---

## Executive Summary

This interim report documents the progress and findings from a three-task end-to-end fraud detection project. The pipeline covers data preprocessing and exploratory analysis, model building and evaluation, and model explainability using SHAP. Two datasets were used throughout: `Fraud_Data.csv` (e-commerce transactions) and `creditcard.csv` (PCA-transformed bank transaction data). The final selected model is XGBoost, evaluated using AUC-PR as the primary metric due to the severe class imbalance in both datasets. SHAP analysis identified `time_since_signup`, `purchase_value`, and `user_tx_count` as the dominant fraud drivers, leading to five concrete business recommendations for fraud prevention.

---

## 1. Task 1 — Data Analysis and Preprocessing

### 1.1 Datasets

| Dataset | Rows | Features | Target | Fraud Rate |
|---------|------|----------|--------|------------|
| `Fraud_Data.csv` | ~151,000 | 11 | `class` | ~9–10% |
| `creditcard.csv` | ~284,807 | 30 (PCA) | `Class` | ~0.17% |
| `IpAddress_to_Country.csv` | ~138,000 IP ranges | 3 | — | — |

### 1.2 Data Cleaning

**Fraud_Data.csv:**
- No significant missing values found. Missing IP addresses were retained and labeled `Unknown` during geolocation merge.
- Duplicates removed with `drop_duplicates()`.
- `signup_time` and `purchase_time` cast from string to `datetime64` — required for all time-based feature derivations.

**creditcard.csv:**
- No missing values (all 30 features are PCA-transformed numeric outputs).
- Duplicates removed.
- No data type corrections required.

### 1.3 Exploratory Data Analysis

**Univariate distributions:**
- `purchase_value` is right-skewed; most transactions are below $100 with a long tail. Fraudulent transactions trend toward higher values.
- `age` follows a roughly bell-shaped distribution centred around 30–40 years.
- Traffic source is dominated by SEO and Ads; Chrome and IE are the most common browsers.
- In `creditcard.csv`, `Amount` is heavily right-skewed and `Time` shows a bimodal distribution consistent with day/night transaction cycles.

**Bivariate analysis:**
- KDE plots confirm fraudulent transactions have a heavier right tail in `purchase_value`.
- Fraud rate by source and browser is broadly comparable, with marginal elevations in Direct traffic and IE browser.
- For `creditcard.csv`, PCA components `V17`, `V14`, `V12`, `V10`, and `V11` show the highest absolute Pearson correlation with the fraud label. `Amount` has surprisingly low direct correlation.

**Class imbalance:**

| Dataset | Legitimate | Fraud | Fraud Rate |
|---------|-----------|-------|------------|
| Fraud_Data.csv | ~136,000 | ~15,000 | ~9–10% |
| creditcard.csv | ~284,315 | ~492 | ~0.17% |

Both datasets are imbalanced; creditcard.csv is severely so, rendering standard accuracy meaningless.

### 1.4 Geolocation Integration

IP addresses were converted to 32-bit integers using the formula:

```
ip_int = a × 16,777,216 + b × 65,536 + c × 256 + d
```

A range-based merge was performed using `pd.merge_asof` (backward direction on sorted arrays), matching each transaction's integer IP against the `lower_bound`/`upper_bound` ranges in `IpAddress_to_Country.csv`. A post-merge validity check (`ip_int <= upper_bound`) filtered out false matches; unmatched IPs were labeled `Unknown`. Fraud rate was then aggregated by country (minimum 100 transactions) to identify geographic risk concentration.

A direct SQL-style join is impossible on range data — `merge_asof` on sorted integer arrays is both correct and O(n log n) efficient.

### 1.5 Feature Engineering

The following features were engineered from `Fraud_Data.csv`:

| Feature | Derivation | Fraud Signal |
|---------|-----------|--------------|
| `hour_of_day` | `purchase_time.dt.hour` | Late-night/early-morning concentration |
| `day_of_week` | `purchase_time.dt.dayofweek` | Weekend activity patterns |
| `time_since_signup` | `(purchase_time − signup_time) / 3600` (hours) | Short gap = synthetic/stolen account |
| `user_tx_count` | Count of all transactions per `user_id` | Low count = new/disposable account |
| `tx_last_24h` | Cumulative expanding count per user (sorted by time) | High count = card testing / burst fraud |

`time_since_signup` is the single most important engineered feature. Legitimate users typically browse and purchase over days or weeks after signup. A gap of minutes to a few hours is a strong indicator of an account created solely for a fraudulent transaction.

### 1.6 Data Transformation

- **StandardScaler**: Fit on training data only; applied to both train and test to prevent leakage.
- **One-hot encoding**: `source`, `browser`, `sex`, `country` encoded with `drop_first=True` to avoid multicollinearity.
- **Dropped columns**: `user_id`, `device_id`, `ip_address`, `ip_int`, `signup_time`, `purchase_time` — replaced by engineered features.

### 1.7 Class Imbalance — SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) was chosen over random undersampling because undersampling would discard ~283,000 legitimate records from `creditcard.csv` (reducing the training set to ~984 rows). SMOTE synthesizes realistic minority-class samples via k-nearest-neighbour interpolation, preserving the full majority class.

SMOTE was applied **exclusively to the training set** after the stratified split to prevent test set contamination.

| Dataset | Before SMOTE (fraud rate) | After SMOTE (fraud rate) |
|---------|--------------------------|--------------------------|
| Fraud_Data train | ~9–10% | 50% |
| creditcard train | ~0.17% | 50% |

The test sets retain original distributions for realistic evaluation.

---

## 2. Task 2 — Model Building and Training

### 2.1 Models Trained

Two models were trained on both datasets:

| Model | Type | Purpose |
|-------|------|---------|
| Logistic Regression | Linear baseline | Interpretable reference; regulatory auditability |
| XGBoost | Gradient boosting ensemble | Primary production model |

### 2.2 Logistic Regression (Baseline)

Configuration: `max_iter=1000`, `random_state=42`, trained on SMOTE-resampled training data.

Logistic Regression provides a linear decision boundary and directly interpretable coefficients. Its limitations — inability to capture non-linear feature interactions such as `time_since_signup × user_tx_count` — are the primary reason it underperforms on both datasets relative to XGBoost.

### 2.3 XGBoost (Ensemble)

Hyperparameter tuning via `GridSearchCV` with `StratifiedKFold(3)` and `scoring='average_precision'`:

| Hyperparameter | Values Searched |
|---------------|----------------|
| `n_estimators` | 100, 200 |
| `max_depth` | 3, 5 |
| `learning_rate` | 0.05, 0.1 |

XGBoost builds trees sequentially, each correcting the residuals of the previous. This allows it to learn non-linear thresholds (e.g., `time_since_signup < 1 hour` as a hard rule) and feature interactions that Logistic Regression requires explicit manual engineering to capture.

### 2.4 Cross-Validation (Stratified K-Fold, k=5)

Cross-validation was run on the raw **pre-SMOTE** datasets to produce unbiased fold-level estimates. Running CV on SMOTE-resampled data would leak synthetic samples across folds and inflate performance estimates.

Each fold used `StratifiedKFold` to preserve the original class ratio within each split. Metrics reported are mean ± standard deviation across 5 folds.

### 2.5 Evaluation Metrics

Given severe class imbalance, three metrics were used:

| Metric | Rationale |
|--------|-----------|
| **AUC-PR** (primary) | Directly measures precision-recall tradeoff; not inflated by true negatives |
| **F1-Score** | Harmonic mean of precision and recall on the fraud class |
| **ROC-AUC** | Supplementary; can be misleadingly high under imbalance |

Raw accuracy was excluded — a classifier predicting all-legitimate achieves 99.83% accuracy on `creditcard.csv` while detecting zero fraud.

### 2.6 Model Comparison and Selection

**Selected model: XGBoost**

XGBoost consistently outperforms Logistic Regression on AUC-PR and F1 across both datasets and in cross-validation. The performance gap is most pronounced on `creditcard.csv` due to its extreme imbalance and the non-linear structure of fraud patterns in the PCA feature space.

| Criterion | Logistic Regression | XGBoost | Winner |
|-----------|-------------------|---------|--------|
| AUC-PR | Lower | Higher | XGBoost |
| F1 Score | Lower | Higher | XGBoost |
| ROC-AUC | Lower | Higher | XGBoost |
| Non-linear patterns | Cannot capture | Yes | XGBoost |
| Interpretability | Native coefficients | Via SHAP | Tie |
| Training speed | Fast | Moderate | LR |

Logistic Regression is retained as the auditable baseline for regulatory reporting. XGBoost interpretability is addressed fully in Task 3 via SHAP.

### 2.7 Saved Model Artifacts

| File | Description |
|------|-------------|
| `models/xgb_fraud.pkl` | Tuned XGBoost — Fraud_Data |
| `models/xgb_creditcard.pkl` | Tuned XGBoost — CreditCard |
| `models/lr_fraud.pkl` | Logistic Regression — Fraud_Data |
| `models/lr_creditcard.pkl` | Logistic Regression — CreditCard |

---

## 3. Task 3 — Model Explainability

### 3.1 Built-in Feature Importance vs SHAP

XGBoost's built-in `feature_importances_` reports total gain per feature across all trees. SHAP (SHapley Additive exPlanations) computes per-prediction signed feature attributions using game-theoretic Shapley values.

**Key differences:**
- Built-in gain is a global aggregate that can be inflated by features used in many shallow splits.
- SHAP measures actual prediction impact per sample — signed (positive = pushes toward fraud), consistent, and additive.
- Both methods agree on the top-3 features for both datasets, validating the feature engineering choices from Task 1.

**Fraud_Data — Top 5 Drivers (SHAP):**

| Rank | Feature | Business Meaning |
|------|---------|-----------------|
| 1 | `time_since_signup` | Accounts transacting within hours of creation are extremely high risk |
| 2 | `purchase_value` | High-value purchases from new/unverified accounts are disproportionately fraudulent |
| 3 | `user_tx_count` | Single-transaction accounts are predominantly fraudulent |
| 4 | `tx_last_24h` | Burst transactions signal card testing or rapid fund extraction |
| 5 | `hour_of_day` | Late-night/early-morning transactions have elevated fraud rates |

**creditcard.csv — Top 5 Drivers (SHAP):**

| Rank | Feature | Business Meaning |
|------|---------|-----------------|
| 1 | `V17` | Strongest PCA-derived behavioral fraud signature |
| 2 | `V14` | Second-order behavioral pattern |
| 3 | `V12` | Complementary fraud pattern — works in combination with V17/V14 |
| 4 | `V10` | Consistent fraud indicator across the distribution |
| 5 | `V11` | Secondary structural pattern in fraud transactions |

### 3.2 SHAP Force Plot Analysis

Three individual predictions were analyzed from the `Fraud_Data` test set:

**True Positive (correctly identified fraud):**  
`time_since_signup` was extremely low (minutes after signup), `user_tx_count` = 1, and `purchase_value` was high. All three signals combined to push the prediction well above the fraud threshold. This is the canonical new-account fraud pattern — account creation velocity abuse.

**False Positive (legitimate flagged as fraud):**  
The transaction had a moderately low `time_since_signup` (a few hours), above-average `purchase_value`, and a high-risk country flag. No single signal was conclusive, but their combination crossed the decision threshold. These cases should be routed to step-up verification (OTP/document check) rather than outright block — the model is uncertain, not certain of fraud.

**False Negative (missed fraud):**  
The fraudster used a "seasoned" account with multiple prior transactions and a signup age of several days. `time_since_signup` and `user_tx_count` both produced negative SHAP values — actively pushing toward legitimate — causing the model to miss the fraud. This is account-takeover fraud, which the current feature set cannot reliably detect without behavioral deviation features.

### 3.3 Counterintuitive Findings

1. **`Amount` is a weak predictor in `creditcard.csv`** — fraud occurs across all transaction sizes because fraudsters deliberately keep amounts small to avoid rule-based thresholds.
2. **`age` interacts non-monotonically with `time_since_signup`** — older users with very short signup times are more suspicious than younger users with the same gap; age alone is not a reliable fraud signal.
3. **The model's primary failure mode is account-takeover fraud** — the false-negative pattern is not random but systematic, reflecting a structural gap in the current feature set.

### 3.4 Business Recommendations

**Recommendation 1 — Signup Velocity Rule**  
Transactions within 1 hour of account signup should be automatically routed to step-up verification (OTP or document check) before processing. SHAP shows a sharp fraud probability threshold below ~1 hour of signup age — this single rule would intercept the majority of new-account fraud cases.

**Recommendation 2 — Single-Transaction Account Holds**  
First-time purchases from accounts with no prior transaction history should be held for 15 minutes with automated velocity checks before fulfillment. `user_tx_count` = 1 consistently produces large positive SHAP values. A 15-minute hold enables real-time cross-account device/IP checks without permanently blocking new customers.

**Recommendation 3 — Geographic Risk Tiering**  
Transactions from IP addresses in high-risk countries (identified via SHAP country feature contributions and EDA fraud-rate-by-country analysis) should require additional verification for purchases above $100. Combining geographic risk with a purchase value threshold creates a more precise two-factor rule than either signal alone.

**Recommendation 4 — Burst Transaction Monitoring**  
Users making more than 3 transactions within a 24-hour window should trigger a real-time review flag, especially if any transaction exceeds their historical average. `tx_last_24h` produces meaningful positive SHAP at high counts, corresponding to card-testing and rapid-extraction fraud patterns.

**Recommendation 5 — Account Takeover Detection Layer**  
Invest in behavioral deviation features — flag transactions where the current session's device fingerprint, IP country, or purchase category differs from the account's historical baseline. The false-negative SHAP analysis shows the model is blind to account-takeover fraud due to the absence of any historical-deviation signal. Adding `is_new_device`, `is_new_country`, and `category_change_flag` features would directly address this gap.

---

## 4. Project Structure and Reproducibility

```
fraud-detection/
├── .github/workflows/unittests.yml   # CI — pytest on push/PR
├── data/
│   ├── raw/                          # Original CSVs (gitignored)
│   └── processed/                    # Cleaned and engineered datasets
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap-explainability.ipynb
├── models/                           # Saved .pkl model artifacts
├── reports/
│   ├── task1_report.md
│   ├── task2_report.md
│   ├── task3_report.md
│   └── interim_report.md
├── src/
├── tests/
└── requirements.txt
```

**Setup:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Run order:** `eda-fraud-data.ipynb` → `eda-creditcard.ipynb` → `feature-engineering.ipynb` → `modeling.ipynb` → `shap-explainability.ipynb`

**Git branches:**

| Branch | Content |
|--------|---------|
| `main` | Full project — all tasks |
| `task-1` | Data cleaning, EDA, feature engineering, SMOTE |
| `task-2` | Modeling, cross-validation, model selection |
| `task-3` | SHAP explainability, business recommendations |

---

## 5. Summary and Next Steps

| Task | Status | Key Output |
|------|--------|-----------|
| Task 1 — Data Preprocessing | Complete | 6 processed CSV files, engineered features, SMOTE splits |
| Task 2 — Model Building | Complete | XGBoost selected; AUC-PR/F1/ROC-AUC evaluated; 4 models saved |
| Task 3 — Explainability | Complete | SHAP summary/force/dependence plots; 5 business recommendations |

**Planned next steps:**
- Deploy the XGBoost model as a REST API (Flask/FastAPI) for real-time scoring
- Add behavioral deviation features (`is_new_device`, `is_new_country`) to address the account-takeover false-negative gap identified by SHAP
- Implement a model monitoring dashboard to track fraud rate drift and feature distribution shift in production
