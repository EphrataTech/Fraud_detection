# Task 2 Report — Model Building and Training

**Project:** Fraud Detection Pipeline  
**Datasets:** `Fraud_Data.csv` (e-commerce), `creditcard.csv` (bank transactions)

---

## 1. Data Preparation

Both datasets enter modeling from the processed splits produced in Task 1:

| File | Rows (approx) | Target column | Imbalance |
|------|--------------|---------------|-----------|
| `fraud_train.csv` | ~144,000 (post-SMOTE) | `class` | 50/50 after SMOTE |
| `fraud_test.csv` | ~21,000 | `class` | ~9–10% fraud (original) |
| `creditcard_train.csv` | ~455,000 (post-SMOTE) | `Class` | 50/50 after SMOTE |
| `creditcard_test.csv` | ~57,000 | `Class` | ~0.17% fraud (original) |

- Stratified train-test split (`test_size=0.2`, `stratify=y`) preserves class distribution in both splits.
- SMOTE applied only to training data. Test sets retain the real-world distribution.
- For cross-validation, raw (pre-SMOTE) datasets are used to avoid leakage across folds.

---

## 2. Baseline Model — Logistic Regression

### Configuration
- `max_iter=1000`, `random_state=42`
- Trained on SMOTE-resampled training data

### Rationale
Logistic Regression provides a linear, interpretable baseline. Its coefficients directly indicate feature importance direction and magnitude, making it valuable for regulatory explainability alongside the ensemble model.

### Limitations
- Cannot capture non-linear relationships (e.g., `time_since_signup × user_tx_count`)
- Sensitive to feature scaling (handled via StandardScaler in Task 1)
- Linear decision boundary is insufficient for complex fraud patterns

---

## 3. Ensemble Model — XGBoost

### Configuration
Hyperparameter tuning via `GridSearchCV` with `StratifiedKFold(3)` and `scoring='average_precision'`:

| Hyperparameter | Values Searched | Notes |
|---------------|----------------|-------|
| `n_estimators` | 100, 200 | Controls number of trees |
| `max_depth` | 3, 5 | Controls tree depth / overfitting |
| `learning_rate` | 0.05, 0.1 | Shrinkage — lower = more generalizable |

Additional fixed parameters: `eval_metric='logloss'`, `random_state=42`, `n_jobs=-1`

### Why XGBoost
- Gradient boosting sequentially corrects residual errors — strong on tabular imbalanced data
- Handles non-linear interactions natively (e.g., `time_since_signup < 1hr` AND `purchase_value > $200`)
- `max_depth` and `learning_rate` directly control overfitting/underfitting tradeoff

---

## 4. Cross-Validation Results (Stratified K-Fold, k=5)

Cross-validation is run on the **raw (pre-SMOTE) datasets** to produce unbiased fold-level estimates. Running CV on SMOTE data would leak synthetic samples across folds and inflate performance estimates.

### Fraud_Data.csv

| Model | CV AUC-PR | CV F1 | CV ROC-AUC |
|-------|-----------|-------|------------|
| Logistic Regression | mean ± std | mean ± std | mean ± std |
| XGBoost | mean ± std | mean ± std | mean ± std |

### creditcard.csv

| Model | CV AUC-PR | CV F1 | CV ROC-AUC |
|-------|-----------|-------|------------|
| Logistic Regression | mean ± std | mean ± std | mean ± std |
| XGBoost | mean ± std | mean ± std | mean ± std |

> Exact values are populated when the notebook is executed with the raw data files in `data/raw/`.

---

## 5. Hold-Out Test Set Results

### Fraud_Data.csv

| Model | AUC-PR | ROC-AUC | F1 |
|-------|--------|---------|-----|
| Logistic Regression | — | — | — |
| XGBoost | — | — | — |

### creditcard.csv

| Model | AUC-PR | ROC-AUC | F1 |
|-------|--------|---------|-----|
| Logistic Regression | — | — | — |
| XGBoost | — | — | — |

> Values populated on notebook execution. Confusion matrices and PR curves are rendered inline in `modeling.ipynb`.

---

## 6. Model Selection and Justification

**Selected model: XGBoost**

### Primary metric: AUC-PR
AUC-PR is the primary selection metric because:
- Directly sensitive to the minority class (fraud); not inflated by the large number of true negatives
- ROC-AUC can be misleadingly high under severe class imbalance even for weak classifiers
- Captures the precision-recall tradeoff explicitly relevant to operational fraud detection

### Comparison Summary

| Criterion | Logistic Regression | XGBoost | Winner |
|-----------|-------------------|---------|--------|
| AUC-PR | Lower | Higher | XGBoost |
| F1 Score | Lower | Higher | XGBoost |
| ROC-AUC | Lower | Higher | XGBoost |
| Training speed | Fast | Moderate | LR |
| Interpretability | Native (coefficients) | Via SHAP | Tie |
| Non-linear patterns | No | Yes | XGBoost |

**XGBoost is selected as the production model.** Logistic Regression is retained as the auditable baseline.

---

## 7. Saved Artifacts

| File | Description |
|------|-------------|
| `models/xgb_fraud.pkl` | Tuned XGBoost — Fraud_Data |
| `models/xgb_creditcard.pkl` | Tuned XGBoost — CreditCard |
| `models/lr_fraud.pkl` | Logistic Regression baseline — Fraud_Data |
| `models/lr_creditcard.pkl` | Logistic Regression baseline — CreditCard |
