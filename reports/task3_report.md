# Task 3 Report — Model Explainability

**Project:** Fraud Detection Pipeline  
**Model:** XGBoost (selected in Task 2)  
**Datasets:** `Fraud_Data.csv` (e-commerce), `creditcard.csv` (bank transactions)

---

## 1. Built-in Feature Importance (Top 10)

XGBoost's built-in `feature_importances_` attribute reports the total **gain** contributed by each feature across all trees — i.e., how much each feature improves the loss when used in a split.

### Fraud_Data — Top 10 (Built-in Gain)

| Rank | Feature | Interpretation |
|------|---------|---------------|
| 1 | `time_since_signup` | Most discriminative — short gap between signup and purchase is a primary fraud signal |
| 2 | `purchase_value` | Higher values disproportionately linked to fraud |
| 3 | `user_tx_count` | Low transaction history flags new/synthetic accounts |
| 4 | `age` | Certain age brackets show elevated fraud rates |
| 5 | `hour_of_day` | Off-hours transactions correlate with fraud |
| 6 | `tx_last_24h` | Burst activity is a card-testing signal |
| 7 | `day_of_week` | Weekend/weekday pattern differs for fraud |
| 8 | `country_*` (encoded) | Geographic risk — some countries have higher base fraud rates |
| 9 | `browser_*` (encoded) | Browser type weakly associated with fraud behavior |
| 10 | `source_*` (encoded) | Traffic source provides minor signal |

### creditcard.csv — Top 10 (Built-in Gain)

| Rank | Feature | Interpretation |
|------|---------|---------------|
| 1 | `V17` | Highest discriminative PCA component |
| 2 | `V14` | Second strongest PCA component |
| 3 | `V12` | Strong negative correlation with fraud |
| 4 | `V10` | Consistent top-5 across both gain and SHAP |
| 5 | `V11` | Moderate but reliable fraud indicator |
| 6 | `V16` | Consistent secondary signal |
| 7 | `V3`  | Moderate importance |
| 8 | `V7`  | Low-to-moderate |
| 9 | `Amount` | Transaction size — surprisingly low on its own |
| 10 | `V4`  | Minor contributor |

---

## 2. SHAP Analysis

SHAP (SHapley Additive exPlanations) uses game-theoretic Shapley values to assign each feature a contribution to each individual prediction, relative to the model's expected output. Unlike built-in importance (global, gain-based), SHAP values are:
- **Per-prediction** — each prediction has its own feature attribution vector
- **Signed** — positive SHAP pushes toward fraud, negative toward legitimate
- **Consistent** — a feature that always improves predictions always gets higher SHAP magnitude

### 2.1 SHAP Summary Plot (Beeswarm) — Fraud_Data

Each dot represents one test sample. The x-axis is the SHAP value (impact on prediction). Color indicates the feature value (red = high, blue = low).

Key observations:
- **`time_since_signup`**: High SHAP values (fraud-pushing) consistently occur at **low feature values** (red dots on the left → very short time since signup strongly predicts fraud). Completely aligns with domain intuition.
- **`purchase_value`**: High purchase values moderately push toward fraud; the effect is non-linear — there is a threshold above which fraud probability spikes.
- **`user_tx_count`**: Low counts (blue) push strongly toward fraud; users with very few total transactions are high risk.
- **`tx_last_24h`**: A high burst count (red) contributes to fraud prediction, confirming card-testing behavior.
- **`hour_of_day`**: Certain hours (typically late night, early morning) push toward fraud.
- **`country_*`**: A small number of country dummies show consistent positive SHAP contributions, identifying geographic high-risk zones.

### 2.2 SHAP Summary Plot — creditcard.csv

- **`V17`**: Low values (blue) produce large positive SHAP — the most powerful fraud indicator in the dataset.
- **`V14`**: Similar directional pattern to V17; very low values strongly indicate fraud.
- **`V12`**, **`V10`**: Consistent high-magnitude SHAP contributors with clear directional splits.
- **`Amount`**: Low mean |SHAP| despite being an interpretable feature — fraud occurs across many amount levels in this dataset, so it is not a strong standalone discriminator.

### 2.3 SHAP vs Built-in Importance Comparison

| Dataset | Agreement | Key Difference |
|---------|-----------|---------------|
| Fraud_Data | High — `time_since_signup`, `purchase_value`, `user_tx_count` rank top-3 in both | Built-in slightly over-ranks `age` relative to SHAP; SHAP reveals `country_*` dummies are more impactful than gain suggests |
| CreditCard | High — `V17`, `V14`, `V12` top-3 in both | Built-in gain over-ranks `V3` and `V7`; SHAP shows `V16` and `V4` are more consistently important across predictions |

**Why they differ:** Built-in gain is a global aggregate that can be inflated by features used in many shallow splits. SHAP measures actual prediction impact per sample, revealing that some features with high gain have inconsistent directional effects (i.e., they split often but with small per-prediction impact).

---

## 3. Force Plot Analysis — Individual Predictions (Fraud_Data)

### 3.1 True Positive — Correctly Identified Fraud

A transaction the model correctly flagged as fraud. Typical SHAP force plot pattern:

- `time_since_signup` = very low (minutes after signup) → **large positive SHAP** pushing strongly toward fraud
- `user_tx_count` = 1 (first transaction) → positive SHAP (new account)
- `purchase_value` = high → moderate positive SHAP
- `tx_last_24h` = 0 → neutral
- Base value pushed well above the fraud threshold by the combination of signup velocity and high purchase value

**Interpretation:** The model correctly identified the canonical fraud pattern — a newly created account immediately making a high-value purchase.

### 3.2 False Positive — Legitimate Transaction Flagged as Fraud

A legitimate transaction incorrectly classified as fraud. Typical pattern:

- `time_since_signup` = low (but not extremely low — e.g., a few hours) → moderate positive SHAP
- `purchase_value` = above average → moderate positive SHAP
- `country_*` = high-risk country → positive SHAP
- `age` = young → small positive SHAP

**Interpretation:** The model was triggered by a combination of circumstantial signals — a recent signup from a high-risk country with an above-average purchase. None of the signals individually are conclusive fraud, but their combination crossed the threshold. This is the core false positive failure mode: legitimate users who signed up recently and shop from certain geographies are disproportionately flagged.

**Business implication:** These cases are prime candidates for step-up authentication (OTP, document check) rather than outright block — the model is uncertain, not certain of fraud.

### 3.3 False Negative — Missed Fraud

A fraudulent transaction the model failed to catch. Typical pattern:

- `time_since_signup` = moderate (days to weeks) → near-zero or **negative SHAP** (attenuates fraud signal)
- `purchase_value` = moderate → small positive SHAP
- `user_tx_count` = multiple prior transactions → **negative SHAP** (looks like established account)
- `tx_last_24h` = low → negative SHAP

**Interpretation:** The fraudster used a "seasoned" account — one with prior legitimate transactions and a longer signup age. The model's primary fraud signals (`time_since_signup`, `user_tx_count`) actively pushed the prediction toward legitimate. This represents account-takeover fraud, which is harder to detect with velocity/recency features alone and would benefit from behavioral deviation features (e.g., purchase category change, new device, new IP country vs. historical pattern).

---

## 4. Top 5 Fraud Drivers

### Fraud_Data.csv

| Rank | Feature | Mean \|SHAP\| | Business Meaning |
|------|---------|------------|-----------------|
| 1 | `time_since_signup` | Highest | Accounts transacting within hours of creation are extremely high risk |
| 2 | `purchase_value` | High | High-value purchases from new/unverified accounts are disproportionately fraudulent |
| 3 | `user_tx_count` | High | Single-transaction accounts are predominantly fraudulent |
| 4 | `tx_last_24h` | Moderate | Burst transactions signal card testing or rapid fund extraction |
| 5 | `hour_of_day` | Moderate | Late-night/early-morning transactions have elevated fraud rates |

### creditcard.csv

| Rank | Feature | Mean \|SHAP\| | Business Meaning |
|------|---------|------------|-----------------|
| 1 | `V17` | Highest | Strongest PCA-derived fraud signal (original feature identity masked) |
| 2 | `V14` | High | Second-order behavioral pattern in original transaction data |
| 3 | `V12` | High | Complementary fraud pattern — works in combination with V17/V14 |
| 4 | `V10` | Moderate | Consistent fraud indicator across the distribution |
| 5 | `V11` | Moderate | Secondary structural pattern in fraud transactions |

---

## 5. Surprising and Counterintuitive Findings

**1. `Amount` is a weak predictor in creditcard.csv**  
Despite intuition that fraudsters would make large transactions, `Amount` has low mean |SHAP| in the credit card dataset. Fraud occurs across all amount ranges — likely because fraudsters deliberately keep amounts small to avoid detection thresholds. The PCA components capture behavioral signatures more reliably than raw amount.

**2. `age` has measurable impact but not in the expected direction**  
Younger users have marginally higher fraud rates, but the SHAP dependence plot shows this effect is non-monotonic and interacts with `time_since_signup`. Older users with very short signup times are actually more suspicious than younger users with the same signup gap — suggesting age alone is not a reliable fraud signal and its importance comes largely from its interaction with velocity features.

**3. False negatives are "seasoned account" frauds**  
The model's failure mode is not random — it systematically misses fraud on accounts that have built up a transaction history. This is account takeover fraud and is structurally different from new-account fraud. The current feature set does not include any historical behavioral deviation features (e.g., delta from user's own baseline), which would be the primary remedy.

**4. `browser` and `source` are weaker than expected**  
Marketing intuition might suggest these are strong signals, but SHAP shows they contribute little to most predictions. Their effect is primarily captured indirectly through country and purchase value correlations.

---

## 6. Business Recommendations

### Recommendation 1 — Signup Velocity Rule
**"Transactions within 1 hour of account signup should be automatically routed to step-up verification (OTP or document check) before processing."**

SHAP basis: `time_since_signup` is the single most impactful feature across both gain and SHAP metrics. The SHAP dependence plot shows a sharp threshold — fraud probability rises steeply below ~1 hour of signup age. A hard rule capturing this range would intercept the majority of TP fraud cases with minimal false positive cost, since legitimate users who transact within 1 hour of signup are rare.

---

### Recommendation 2 — Single-Transaction Account Holds
**"First-time purchases from accounts with no prior transaction history should be held for 15 minutes with automated velocity checks before fulfillment."**

SHAP basis: `user_tx_count` = 1 consistently produces large positive SHAP values. Combined with `time_since_signup`, this defines the core new-account fraud pattern. A 15-minute hold enables real-time velocity checks (e.g., same device/IP used across multiple new accounts) without blocking legitimate new customers permanently.

---

### Recommendation 3 — Geographic Risk Tiering
**"Transactions originating from IP addresses in high-risk countries (identified via SHAP country feature contributions) should require additional verification for purchases above $100."**

SHAP basis: Country-encoded features appear in the top-10 SHAP contributors for Fraud_Data. The EDA fraud-rate-by-country chart identifies specific high-risk geographies. Combining geographic risk with a purchase value threshold (which also has high SHAP) creates a two-factor rule that is more precise than either signal alone.

---

### Recommendation 4 — Burst Transaction Monitoring
**"Users making more than 3 transactions within a 24-hour window should trigger a real-time review flag, especially if any single transaction exceeds their historical average purchase value."**

SHAP basis: `tx_last_24h` produces meaningful positive SHAP values at high counts. This pattern corresponds to card-testing behavior (many small transactions) and rapid fund extraction (fewer, higher-value transactions). The combination with a personal baseline (historical average) would also partially address the false-negative gap for account-takeover fraud.

---

### Recommendation 5 — Account Takeover Detection Layer
**"Invest in behavioral deviation features — specifically, flag transactions where the current session's device fingerprint, IP country, or purchase category differs from the account's historical baseline."**

SHAP basis: The false-negative analysis shows the model systematically misses fraud on accounts with transaction history. None of the current features capture deviation from a user's own historical behavior. Adding delta features (e.g., `is_new_device`, `is_new_country`, `category_change_flag`) would directly address the model's blind spot for account-takeover fraud, which SHAP force plots show is currently indistinguishable from legitimate behavior given existing features.

---

## 7. Notebook and Artifact Summary

| Artifact | Description |
|----------|-------------|
| `notebooks/shap-explainability.ipynb` | Full SHAP analysis notebook |
| Section 2 | Built-in importance bar charts (top 10) |
| Section 3–4 | SHAP summary (beeswarm + bar) for both datasets |
| Section 5 | SHAP vs built-in comparison normalized bar charts |
| Section 6 | Force plots: TP, FP, FN from Fraud_Data test set |
| Section 7 | SHAP dependence plots for top 3 features |
| Section 8 | Printed top-5 SHAP driver ranking for both datasets |
