# Credit Card Spend Forecasting

Predicting average monthly credit card spend for Q3 (July–September) using Q1–Q2 transaction history. Built for the TVS Motor Company x Analytics Vidhya DS Bootcamp hackathon, 2023. Ranked 1st in batch.

**Live page:** [sm1else-bot.github.io/cc-prediction](https://sm1else-bot.github.io/cc-prediction)

---

## Problem

Given three months of a customer's financial activity (credit card spend, debit transactions, bank flows, loan status, investments), predict their average monthly credit card spend for the following quarter. Evaluation metric: RMSLE x 100. Competition threshold: 125. Winning score: 116.12.

---

## Dataset

- 32,820 training rows, 14,067 test rows
- 44 raw features: CC spend (Apr/May/Jun), debit card spend, bank account flows, EMI, active loans, investment holdings, demographics
- Target: `cc_cons` — average monthly CC spend for Jul–Sep
- Source: proprietary TVS Motor Company / Analytics Vidhya dataset, not redistributable

---

## Approach

### Feature engineering (`better_solution.py`)
92 features derived from 44 raw inputs:
- Log-space CC features: `log_cc_apr/may/jun`, `log_cc_avg`, log-space trend and momentum
- Momentum and acceleration: `cc_mom1`, `cc_mom2`, `cc_accel`, `cc_trend`
- Utilisation ratios: `cc_util`, `cc_util_jun`, `log_cc_util`
- Cross-asset ratios: log CC-to-debit ratio, debit momentum
- Target encoding: `region_code` (348 unique values) encoded per fold to prevent leakage
- Coefficient of variation, rolling std in both raw and log space

### Model
5-fold out-of-fold stacking ensemble:
- **LightGBM** (CPU) — OOF RMSLE: 116.89
- **XGBoost** (GPU, `device='cuda'`) — OOF RMSLE: 116.66
- **CatBoost** (GPU, `task_type='GPU'`) — OOF RMSLE: 116.30
- **Ridge meta-learner** on OOF predictions — final OOF RMSLE: **116.28**

CatBoost hyperparameters tuned with Optuna (30 trials, TPE sampler, 3-fold CV).

### Why the score plateaus
R² on this dataset is approximately 0.16. With only three months of transaction history, high missingness in loan and investment features, and no behavioural signals beyond spend amounts, every architecture tested — from linear regression to GPU-accelerated gradient boosting — converged to the same performance band. The data is the constraint, not the model.

---

## Repository structure

```
better_solution.py     Full ML pipeline: feature engineering, OOF stacking, submission
solution.py            Original 2023 submission (stacked sklearn models, XGBoost)
serve.py               Flask server for local model inference (legacy, no longer used)
index.html             Project page with model explanation and scenario table
catboost_model.joblib  Trained CatBoost model + feature names
better_submission.csv  Final submission file
```

---

## Running locally

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna joblib flask flask-cors tqdm
python better_solution.py   # requires train.csv and test_9K3DBWQ_2aRGUxy.csv
```

Training takes approximately 15–20 minutes on a GPU (tested on RTX 4070). LightGBM runs on CPU; XGBoost and CatBoost use CUDA.

---

## Results

| Model | OOF RMSLE x100 |
|---|---|
| LightGBM | 116.89 |
| XGBoost | 116.66 |
| CatBoost | 116.30 |
| Ridge meta (final) | **116.28** |
| Original 2023 submission | 116.12 |

The 2026 rebuild did not improve on the 2023 score despite a significantly more sophisticated pipeline. See the Author's Note on the live page.

---

Jessenth Ebenezer — [jessenth.com](https://jessenth.com)
