# Task 1 Report — Data Analysis and Preprocessing

**Project:** Fraud Detection Pipeline  
**Datasets:** `Fraud_Data.csv` (e-commerce), `creditcard.csv` (bank transactions), `IpAddress_to_Country.csv`

---

## 1. Data Cleaning and Preprocessing

### 1.1 Fraud_Data.csv

| Step | Action | Justification |
|------|--------|---------------|
| Missing values | Checked per column | No significant nulls; missing IPs labeled `Unknown` country |
| Duplicates | `drop_duplicates()` applied | Duplicate rows skew frequency-based features and model training |
| Data types | `signup_time`, `purchase_time` cast to `datetime64` | Required for time-delta calculations |

**Columns after cleaning:**

| Column | Type | Notes |
|--------|------|-------|
| `user_id`, `device_id` | object | Dropped before modeling |
| `signup_time`, `purchase_time` | datetime64 | Used to derive time features |
| `purchase_value`, `age` | float64 / int64 | Scaled with StandardScaler |
| `ip_address` | object | Converted to integer for range lookup |
| `source`, `browser`, `sex` | object | One-hot encoded |
| `class` | int64 | Target variable |

### 1.2 creditcard.csv

| Step | Action |
|------|--------|
| Missing values | None found — all 30 features are numeric PCA outputs |
| Duplicates | Removed with `drop_duplicates()` |
| Data types | All columns already float64; no correction needed |

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Fraud_Data.csv — Univariate Distributions

**Numerical features:**
- `purchase_value`: Right-skewed; most transactions cluster below $100 with a long tail. Fraudulent transactions tend toward slightly higher values.
- `age`: Roughly bell-shaped, centred around 30–40 years. No extreme outliers.

**Categorical features:**
- `source`: SEO and Ads dominate traffic; Direct is a smaller segment.
- `browser`: Chrome and IE are most common; Safari and FireFox smaller shares.
- `sex`: Approximately balanced between Male and Female users.

### 2.2 Fraud_Data.csv — Bivariate Analysis

- **Purchase value vs class**: KDE plots show fraudulent transactions have a heavier right tail — higher-value purchases are proportionally more fraudulent.
- **Fraud rate by source**: Comparable across sources; slight elevation in Direct traffic.
- **Fraud rate by browser**: Certain browsers (e.g., IE) show marginally elevated fraud rates, possibly reflecting older/less-secure environments.
- **Fraud rate by sex**: Minimal difference — sex is likely a weak predictor.

### 2.3 creditcard.csv — Univariate & Bivariate

- **Amount**: Heavily right-skewed; most transactions are small. Fraudulent transactions do not necessarily involve large amounts.
- **Time**: Bimodal distribution across the 48-hour window, consistent with day/night cycles.
- **PCA features vs Class**: Top features by absolute Pearson correlation with the fraud label: `V17`, `V14`, `V12`, `V10`, `V11`. `Amount` and `Time` have low direct correlation with fraud.

### 2.4 Class Imbalance

| Dataset | Legitimate | Fraud | Fraud Rate |
|---------|-----------|-------|------------|
| Fraud_Data.csv | ~90,000+ | ~15,000 | ~9–10% |
| creditcard.csv | ~284,315 | ~492 | ~0.17% |

Both datasets are imbalanced. The credit card dataset is severely imbalanced (<0.2% fraud), making standard accuracy a misleading metric.

---

## 3. Feature Engineering

All features below are created in `feature-engineering.ipynb` on `Fraud_Data.csv`.

### 3.1 Time-Based Features

| Feature | Derivation | Rationale |
|---------|-----------|-----------|
| `hour_of_day` | `purchase_time.dt.hour` | Fraud often concentrates at unusual hours (late night/early morning) when monitoring is reduced |
| `day_of_week` | `purchase_time.dt.dayofweek` | Weekend transactions may carry different fraud patterns due to lower oversight |
| `time_since_signup` | `(purchase_time − signup_time).total_seconds() / 3600` | Key fraud signal — see below |

#### Why `time_since_signup` is a key feature

Legitimate users browse, compare, and purchase over days or weeks after signup. A `time_since_signup` of minutes to a few hours strongly indicates a synthetic or stolen account created solely for a single fraudulent transaction. This feature directly captures account creation velocity abuse — one of the most reliable behavioral signals in e-commerce fraud.

### 3.2 Transaction Velocity Features

| Feature | Derivation | Rationale |
|---------|-----------|-----------|
| `user_tx_count` | `groupby('user_id').transform('count')` | Total transactions per user — low count on first purchase correlates with fraud |
| `tx_last_24h` | Expanding cumulative count per user sorted by time | Burst activity (many transactions in a short window) is a classic fraud signal for card testing and account takeover |

### 3.3 IP-to-Country Mapping

**Why it matters:** Geographic origin provides context on whether a transaction comes from a high-risk or inconsistent location relative to the user profile.

**Implementation:**

1. **IP to integer conversion** — Each dotted-decimal IP `a.b.c.d` is converted as:
   ```
   ip_int = a × 16,777,216 + b × 65,536 + c × 256 + d
   ```
   This enables numerical range comparisons against the lookup table.

2. **Range-based merge with `merge_asof`** — A standard join is impossible because the mapping defines IP *ranges* (`lower_bound`, `upper_bound`), not exact IPs. `pd.merge_asof` performs a backward nearest-key merge on sorted arrays, finding the closest lower bound efficiently. A validity check (`ip_int <= upper_bound`) ensures only true range matches are kept; unmatched IPs are labeled `Unknown`.

3. **Fraud rate by country** — After merging, aggregating fraud rate per country (filtered to ≥100 transactions) reveals geographic concentration of fraud, which the model learns via the one-hot encoded `country` feature.

**Why not a direct join?** A lookup table with millions of ranges cannot be joined on exact keys. `merge_asof` on sorted integer arrays is both correct and O(n log n) efficient.

### 3.4 Encoding and Scaling

| Transformation | Applied To | Method |
|---------------|-----------|--------|
| One-hot encoding | `source`, `browser`, `sex`, `country` | `pd.get_dummies(drop_first=True)` — drops one category to avoid multicollinearity |
| StandardScaler | All numerical features | `fit` on train only, `transform` on both — prevents data leakage |
| Dropped columns | `user_id`, `device_id`, `ip_address`, `ip_int`, `signup_time`, `purchase_time` | High-cardinality identifiers and raw timestamps replaced by engineered features |

---

## 4. Class Imbalance Handling Strategy

### 4.1 Chosen Technique: SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) was selected over random undersampling for both datasets.

**How SMOTE works:** For each minority-class sample, SMOTE selects k nearest neighbours and synthesizes new samples by interpolating along the line segments connecting them — creating realistic fraud examples rather than exact duplicates.

### 4.2 Justification

| Criterion | SMOTE | Random Undersampling |
|-----------|-------|---------------------|
| Information loss | None — all majority samples retained | High — discards real majority data |
| Dataset size after resampling | Grows to balanced | Shrinks significantly |
| Risk of overfitting | Low (synthetic, not duplicate) | Low |
| Suitable for severe imbalance | Yes | Risky — loses too much data |
| Applied to test set | No | No |

For `creditcard.csv` with a 0.17% fraud rate, undersampling would reduce the training set to ~984 rows (492 fraud × 2), discarding ~283,000 legitimate examples. SMOTE preserves all legitimate data while balancing the minority class.

### 4.3 Class Distribution Before and After SMOTE

**Fraud_Data.csv (training set):**

| Class | Before SMOTE | After SMOTE |
|-------|-------------|-------------|
| 0 — Legitimate | ~72,000 | ~72,000 |
| 1 — Fraud | ~12,000 | ~72,000 |
| Fraud rate | ~9–10% | 50% |

**creditcard.csv (training set):**

| Class | Before SMOTE | After SMOTE |
|-------|-------------|-------------|
| 0 — Legitimate | ~227,451 | ~227,451 |
| 1 — Fraud | ~394 | ~227,451 |
| Fraud rate | ~0.17% | 50% |

> SMOTE is applied **exclusively on the training set** after the train/test split (`test_size=0.2`, `stratify=y`). The test set retains the original real-world class distribution to ensure evaluation metrics reflect true deployment conditions.

### 4.4 Evaluation Implications

Because the test set remains imbalanced, model performance must be measured with:
- **Precision-Recall AUC** — preferred for imbalanced fraud detection
- **F1-score** on the fraud class
- **ROC-AUC**

Raw accuracy should be avoided — a naive all-legitimate classifier achieves 99.83% accuracy on `creditcard.csv` while detecting zero fraud.

---

## 5. Output Artifacts

| File | Description |
|------|-------------|
| `data/processed/fraud_with_country.csv` | Fraud_Data enriched with `country` from IP lookup |
| `data/processed/fraud_train.csv` | Fraud_Data train split — scaled + SMOTE resampled |
| `data/processed/fraud_test.csv` | Fraud_Data test split — scaled, original distribution |
| `data/processed/creditcard_cleaned.csv` | Deduplicated creditcard dataset |
| `data/processed/creditcard_train.csv` | Creditcard train split — scaled + SMOTE resampled |
| `data/processed/creditcard_test.csv` | Creditcard test split — scaled, original distribution |
