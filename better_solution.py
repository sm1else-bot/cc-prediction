import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5

# ──────────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────────
train = pd.read_csv("train.csv")
test  = pd.read_csv("test_9K3DBWQ_2aRGUxy.csv")
print(f"Train: {train.shape}  Test: {test.shape}")
train = train[train['age'] < 100].reset_index(drop=True)

# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────
def engineer(df):
    d = df.copy()

    zero_fill = ['dc_cons_apr','dc_cons_may','dc_cons_jun',
                 'dc_count_apr','dc_count_may','dc_count_jun',
                 'debit_amount_apr','debit_amount_may','debit_amount_jun',
                 'credit_amount_apr','credit_amount_may','credit_amount_jun',
                 'debit_count_apr','debit_count_may','debit_count_jun',
                 'credit_count_apr','credit_count_may','credit_count_jun',
                 'max_credit_amount_apr','max_credit_amount_may','max_credit_amount_jun',
                 'cc_count_apr','cc_count_may','cc_count_jun',
                 'investment_1','investment_2','investment_3','investment_4',
                 'personal_loan_active','vehicle_loan_active',
                 'personal_loan_closed','vehicle_loan_closed',
                 'emi_active']
    for c in zero_fill:
        if c in d.columns:
            d[c] = d[c].fillna(0)
    d['card_lim'] = d['card_lim'].fillna(d['card_lim'].median())

    apr, may, jun = d['cc_cons_apr'], d['cc_cons_may'], d['cc_cons_jun']

    # Raw CC features
    d['cc_avg']   = (apr + may + jun) / 3
    d['cc_trend'] = jun - apr
    d['cc_mom1']  = may - apr
    d['cc_mom2']  = jun - may
    d['cc_accel'] = d['cc_mom2'] - d['cc_mom1']
    d['cc_std']   = d[['cc_cons_apr','cc_cons_may','cc_cons_jun']].std(axis=1)
    d['cc_min']   = d[['cc_cons_apr','cc_cons_may','cc_cons_jun']].min(axis=1)
    d['cc_max']   = d[['cc_cons_apr','cc_cons_may','cc_cons_jun']].max(axis=1)
    d['cc_cv']    = d['cc_std'] / d['cc_avg'].clip(1)

    # Log-space CC features (critical for RMSLE)
    lapr = np.log1p(apr); lmay = np.log1p(may); ljun = np.log1p(jun)
    d['log_cc_apr']   = lapr
    d['log_cc_may']   = lmay
    d['log_cc_jun']   = ljun
    d['log_cc_avg']   = (lapr + lmay + ljun) / 3
    d['log_cc_trend'] = ljun - lapr
    d['log_cc_mom1']  = lmay - lapr
    d['log_cc_mom2']  = ljun - lmay
    d['log_cc_accel'] = (ljun - lmay) - (lmay - lapr)
    d['log_cc_std']   = d[['log_cc_apr','log_cc_may','log_cc_jun']].std(axis=1)
    d['log_cc_cv']    = d['log_cc_std'] / d['log_cc_avg'].clip(0.01)

    d['log_ratio_jun_apr'] = ljun - lapr
    d['log_ratio_jun_may'] = ljun - lmay
    d['log_ratio_may_apr'] = lmay - lapr

    # Card limit
    d['log_card_lim']  = np.log1p(d['card_lim'])
    d['cc_util']       = d['cc_avg'] / d['card_lim'].clip(1)
    d['cc_util_jun']   = jun / d['card_lim'].clip(1)
    d['log_cc_util']   = d['log_cc_avg'] - d['log_card_lim']

    # CC transaction counts
    d['cc_count_avg']   = (d['cc_count_apr'] + d['cc_count_may'] + d['cc_count_jun']) / 3
    d['cc_count_trend'] = d['cc_count_jun'] - d['cc_count_apr']
    d['cc_avg_txn']     = d['cc_avg'] / d['cc_count_avg'].clip(1)
    d['log_cc_avg_txn'] = np.log1p(d['cc_avg_txn'])

    # Debit card
    dc_apr = d['dc_cons_apr']; dc_may = d['dc_cons_may']; dc_jun = d['dc_cons_jun']
    d['dc_avg']       = (dc_apr + dc_may + dc_jun) / 3
    d['log_dc_avg']   = np.log1p(d['dc_avg'])
    d['log_cc_to_dc'] = d['log_cc_avg'] - np.log1p(d['dc_avg'])
    d['has_dc']       = (d['dc_avg'] > 0).astype(int)

    # Bank flows
    d['avg_credit_amount'] = (d['credit_amount_apr'] + d['credit_amount_may'] + d['credit_amount_jun']) / 3
    d['avg_debit_amount']  = (d['debit_amount_apr']  + d['debit_amount_may']  + d['debit_amount_jun']) / 3
    d['avg_max_credit']    = (d['max_credit_amount_apr'] + d['max_credit_amount_may'] + d['max_credit_amount_jun']) / 3
    d['log_avg_credit']    = np.log1p(d['avg_credit_amount'])
    d['log_avg_debit']     = np.log1p(d['avg_debit_amount'])
    d['log_avg_max_credit']= np.log1p(d['avg_max_credit'])
    d['net_flow']          = d['avg_credit_amount'] - d['avg_debit_amount']
    d['credit_trend']      = d['credit_amount_jun'] - d['credit_amount_apr']

    # Investments
    d['total_investment'] = (d['investment_1'] + d['investment_2'] +
                             d['investment_3'] + d['investment_4'])
    d['has_investment']   = (d['total_investment'] > 0).astype(int)
    d['log_investment']   = np.log1p(d['total_investment'])

    # Loans & EMI
    d['total_active_loans'] = d['personal_loan_active'] + d['vehicle_loan_active']
    d['total_closed_loans'] = d['personal_loan_closed'] + d['vehicle_loan_closed']
    d['has_active_loan']    = (d['total_active_loans'] > 0).astype(int)
    d['log_emi_active']     = np.log1p(d['emi_active'])
    d['has_emi']            = (d['emi_active'] > 0).astype(int)
    d['loan_enq_flag']      = d['loan_enq'].notna().astype(int)

    # Demographics
    d['gender_enc']  = (d['gender'] == 'F').astype(int)
    d['ac_type_enc'] = (d['account_type'] == 'saving').astype(int)

    return d

train = engineer(train)
test  = engineer(test)

drop_cols = {'id', 'cc_cons', 'account_type', 'gender', 'loan_enq'}
feat_cols  = [c for c in train.columns if c not in drop_cols]
print(f"Total features: {len(feat_cols)}")

y      = train['cc_cons'].values
y_log  = np.log1p(y)
X      = train[feat_cols]
X_test = test[feat_cols]

def rmsle100(yt, yp):
    return np.sqrt(mean_squared_log_error(yt, np.maximum(yp, 0))) * 100

global_region_mean = np.log1p(train.groupby('region_code')['cc_cons'].mean())
global_mean_log    = y_log.mean()

# ──────────────────────────────────────────────
# 3. CATBOOST PARAMS (pre-tuned via Optuna)
# ──────────────────────────────────────────────
best_cat_params = {
    'learning_rate':         0.03662819914157703,
    'depth':                 7,
    'l2_leaf_reg':           1.8671503347054141,
    'subsample':             0.9251468899482872,
    'min_data_in_leaf':      37,
    'iterations':            5000,
    'task_type':             'GPU',
    'bootstrap_type':        'Bernoulli',
    'random_seed':           SEED,
    'eval_metric':           'RMSE',
    'early_stopping_rounds': 150,
    'verbose':               False,
}
print("Using pre-tuned CatBoost params (best 3-fold score: 116.7738)")

# ──────────────────────────────────────────────
# 4. OOF STACKING (5-fold)
# ──────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb  = np.zeros(len(X)); test_lgb  = np.zeros(len(X_test))
oof_xgb  = np.zeros(len(X)); test_xgb  = np.zeros(len(X_test))
oof_cat  = np.zeros(len(X)); test_cat  = np.zeros(len(X_test))

fold_bar = tqdm(enumerate(kf.split(X)), total=N_FOLDS, desc="Folds")

for fold, (tr_idx, val_idx) in fold_bar:
    X_tr  = X.iloc[tr_idx].copy()
    X_val = X.iloc[val_idx].copy()
    X_te  = X_test.copy()
    y_tr, y_val_log = y_log[tr_idx], y_log[val_idx]
    y_val_orig = y[val_idx]

    region_mean = np.log1p(train.iloc[tr_idx].groupby('region_code')['cc_cons'].mean())
    X_tr['region_enc']  = X_tr['region_code'].map(region_mean).fillna(global_mean_log)
    X_val['region_enc'] = X_val['region_code'].map(region_mean).fillna(global_mean_log)
    X_te['region_enc']  = X_te['region_code'].map(global_region_mean).fillna(global_mean_log)
    X_tr  = X_tr.drop(columns=['region_code'])
    X_val = X_val.drop(columns=['region_code'])
    X_te  = X_te.drop(columns=['region_code'])

    # ── LightGBM (GPU) ──
    fold_bar.set_description(f"Fold {fold+1}/{N_FOLDS} [LGB]")
    lgb_params = {
        'objective':         'regression',
        'metric':            'rmse',

        'learning_rate':     0.03,
        'num_leaves':        127,
        'min_child_samples': 30,
        'feature_fraction':  0.7,
        'bagging_fraction':  0.8,
        'bagging_freq':      5,
        'reg_alpha':         0.05,
        'reg_lambda':        1.0,
        'verbose':           -1,
        'seed':              SEED,
    }
    lgb_model = lgb.train(
        lgb_params,
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=5000,
        valid_sets=[lgb.Dataset(X_val, label=y_val_log)],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(False)]
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb        += lgb_model.predict(X_te) / N_FOLDS

    # ── XGBoost (GPU) ──
    fold_bar.set_description(f"Fold {fold+1}/{N_FOLDS} [XGB]")
    xgb_model = xgb.XGBRegressor(
        n_estimators=5000, learning_rate=0.03, max_depth=6,
        min_child_weight=30, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.05, reg_lambda=1.0, random_state=SEED,
        device='cuda', eval_metric='rmse',
        early_stopping_rounds=150, verbosity=0
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val_log)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_xgb        += xgb_model.predict(X_te) / N_FOLDS

    # ── CatBoost (GPU, tuned) ──
    fold_bar.set_description(f"Fold {fold+1}/{N_FOLDS} [CAT]")
    cat_model = CatBoostRegressor(**best_cat_params)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val_log))
    oof_cat[val_idx] = cat_model.predict(X_val)
    test_cat        += cat_model.predict(X_te) / N_FOLDS

    s_lgb = rmsle100(y_val_orig, np.expm1(oof_lgb[val_idx]))
    s_xgb = rmsle100(y_val_orig, np.expm1(oof_xgb[val_idx]))
    s_cat = rmsle100(y_val_orig, np.expm1(oof_cat[val_idx]))
    fold_bar.set_postfix({'LGB': f'{s_lgb:.2f}', 'XGB': f'{s_xgb:.2f}', 'CAT': f'{s_cat:.2f}'})

# ──────────────────────────────────────────────
# 5. SCORES & META
# ──────────────────────────────────────────────
print("\n\n=== Full OOF scores (RMSLE*100) ===")
s_lgb_oof = rmsle100(y, np.expm1(oof_lgb))
s_xgb_oof = rmsle100(y, np.expm1(oof_xgb))
s_cat_oof = rmsle100(y, np.expm1(oof_cat))
print(f"LightGBM  OOF: {s_lgb_oof:.4f}")
print(f"XGBoost   OOF: {s_xgb_oof:.4f}")
print(f"CatBoost  OOF: {s_cat_oof:.4f}")

meta = Ridge(alpha=1.0, positive=True)
oof_stack  = np.column_stack([oof_lgb,  oof_xgb,  oof_cat])
test_stack = np.column_stack([test_lgb, test_xgb, test_cat])
meta.fit(oof_stack, y_log)
oof_meta  = meta.predict(oof_stack)
test_meta = meta.predict(test_stack)

s_meta = rmsle100(y, np.expm1(oof_meta))
s_avg  = rmsle100(y, np.expm1((oof_lgb + oof_xgb + oof_cat) / 3))
print(f"Ridge meta OOF:   {s_meta:.4f}  weights=({meta.coef_[0]:.3f},{meta.coef_[1]:.3f},{meta.coef_[2]:.3f})")
print(f"Simple avg OOF:   {s_avg:.4f}")

scores = {'lgb': s_lgb_oof, 'xgb': s_xgb_oof, 'cat': s_cat_oof, 'meta': s_meta, 'avg': s_avg}
best_name  = min(scores, key=scores.get)
best_score = scores[best_name]

test_preds_map = {
    'lgb':  test_lgb,  'xgb': test_xgb, 'cat': test_cat,
    'meta': test_meta, 'avg': (test_lgb + test_xgb + test_cat) / 3,
}
final_log = test_preds_map[best_name]
print(f"\nBest: {best_name.upper()} OOF={best_score:.4f}  (original: 116.12)")

# ──────────────────────────────────────────────
# 6. SAVE MODELS (for serve.py demo)
# ──────────────────────────────────────────────
import joblib

# --- 6a. Full model (for reference) ---
joblib.dump({'model': cat_model, 'feature_names': list(X_tr.columns)}, 'catboost_model.joblib')
print("Saved catboost_model.joblib")

# --- 6b. Demo model: trained only on features the frontend form collects ---
# This guarantees coherent predictions without missing-feature imputation issues.
DEMO_FEATURES = [
    'cc_cons_apr', 'cc_cons_may', 'cc_cons_jun',
    'log_cc_apr', 'log_cc_may', 'log_cc_jun',
    'cc_avg', 'log_cc_avg',
    'cc_trend', 'log_cc_trend',
    'cc_mom1', 'cc_mom2', 'log_cc_mom1', 'log_cc_mom2',
    'cc_accel', 'cc_std', 'cc_cv',
    'card_lim', 'log_card_lim', 'cc_util', 'cc_util_jun', 'log_cc_util',
    'emi_active', 'log_emi_active', 'has_emi',
    'total_active_loans', 'has_active_loan',
    'personal_loan_active', 'vehicle_loan_active',
    'age', 'gender_enc', 'ac_type_enc',
]

X_demo = train[DEMO_FEATURES]
demo_model = CatBoostRegressor(
    iterations=3000, learning_rate=0.05, depth=6,
    l2_leaf_reg=3, task_type='GPU', bootstrap_type='Bernoulli',
    random_seed=SEED, eval_metric='RMSE',
    early_stopping_rounds=100, verbose=False
)

# Quick OOF score for the demo model
oof_demo = np.zeros(len(X_demo))
for tr_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=SEED).split(X_demo):
    dm = CatBoostRegressor(
        iterations=3000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        task_type='GPU', bootstrap_type='Bernoulli',
        random_seed=SEED, eval_metric='RMSE', early_stopping_rounds=100, verbose=False
    )
    dm.fit(X_demo.iloc[tr_idx], y_log[tr_idx],
           eval_set=(X_demo.iloc[val_idx], y_log[val_idx]))
    oof_demo[val_idx] = dm.predict(X_demo.iloc[val_idx])

print(f"Demo model OOF RMSLE*100: {rmsle100(y, np.expm1(oof_demo)):.4f}")

# Train final demo model on full data
demo_model.fit(X_demo, y_log)
joblib.dump({'model': demo_model, 'feature_names': DEMO_FEATURES}, 'catboost_demo_model.joblib')
print("Saved catboost_demo_model.joblib")

# ──────────────────────────────────────────────
# 7. SUBMISSION
# ──────────────────────────────────────────────
final_preds = np.maximum(np.expm1(final_log), 0)
pd.DataFrame({'id': test['id'], 'cc_cons': final_preds}).to_csv('better_submission.csv', index=False)
print(f"Saved better_submission.csv  (mean={final_preds.mean():.0f}, max={final_preds.max():.0f})")
