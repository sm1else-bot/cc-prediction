"""
serve.py — local model server for the credit card spend forecasting demo.

Usage:
    python serve.py

Then open index.html in your browser.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__, static_folder='.')
CORS(app)

MODEL_PATH = 'catboost_demo_model.joblib'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found.\n"
        "Run better_solution.py first to train and save the model."
    )

bundle        = joblib.load(MODEL_PATH)
model         = bundle['model']
feature_names = bundle['feature_names']
print(f"Demo model loaded ({len(feature_names)} features)")


def build_features(d: dict) -> dict:
    """
    Build only the features we can derive from the demo form inputs.
    Everything else will be filled from training medians in the predict endpoint.
    """
    apr = float(d.get('cc_cons_apr', 0))
    may = float(d.get('cc_cons_may', 0))
    jun = float(d.get('cc_cons_jun', 0))
    card_lim = max(float(d.get('card_lim', 1)), 1)
    emi = float(d.get('emi_active', 0))
    pl_active = float(d.get('personal_loan_active', 0))
    vl_active = float(d.get('vehicle_loan_active', 0))

    lapr = np.log1p(apr); lmay = np.log1p(may); ljun = np.log1p(jun)
    cc_avg = (apr + may + jun) / 3
    cc_std_val = float(np.std([apr, may, jun]))

    # Only return features we can actually compute from the form.
    # Unknown features (debit, bank flows, investments, cc counts)
    # are intentionally omitted so the median fallback is used instead.
    return {
        'cc_cons_apr':       apr,
        'cc_cons_may':       may,
        'cc_cons_jun':       jun,
        'cc_avg':            cc_avg,
        'cc_trend':          jun - apr,
        'cc_mom1':           may - apr,
        'cc_mom2':           jun - may,
        'cc_accel':          (jun - may) - (may - apr),
        'cc_std':            cc_std_val,
        'cc_min':            min(apr, may, jun),
        'cc_max':            max(apr, may, jun),
        'cc_cv':             cc_std_val / max(cc_avg, 1),
        'log_cc_apr':        lapr,
        'log_cc_may':        lmay,
        'log_cc_jun':        ljun,
        'log_cc_avg':        (lapr + lmay + ljun) / 3,
        'log_cc_trend':      ljun - lapr,
        'log_cc_mom1':       lmay - lapr,
        'log_cc_mom2':       ljun - lmay,
        'log_cc_accel':      (ljun - lmay) - (lmay - lapr),
        'log_cc_std':        float(np.std([lapr, lmay, ljun])),
        'log_cc_cv':         float(np.std([lapr, lmay, ljun])) / max((lapr + lmay + ljun) / 3, 0.01),
        'log_ratio_jun_apr': ljun - lapr,
        'log_ratio_jun_may': ljun - lmay,
        'log_ratio_may_apr': lmay - lapr,
        'card_lim':          card_lim,
        'log_card_lim':      np.log1p(card_lim),
        'cc_util':           cc_avg / card_lim,
        'cc_util_jun':       jun / card_lim,
        'log_cc_util':       np.log1p(cc_avg) - np.log1p(card_lim),
        'personal_loan_active': pl_active,
        'vehicle_loan_active':  vl_active,
        'total_active_loans':   pl_active + vl_active,
        'has_active_loan':      int((pl_active + vl_active) > 0),
        'emi_active':           emi,
        'log_emi_active':       np.log1p(emi),
        'has_emi':              int(emi > 0),
        'age':                  float(d.get('age', 34)),
        'gender_enc':           1 if d.get('gender') == 'F' else 0,
        'ac_type_enc':          1 if d.get('account_type') == 'saving' else 0,
        'region_enc':           8.02,  # global mean of log1p(cc_cons)
    }


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    try:
        feats = build_features(data)
        X = np.array([[feats[f] for f in feature_names]])
        log_pred = model.predict(X)[0]
        prediction = float(np.expm1(log_pred))
        prediction = max(prediction, 0)
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    print("Starting model server on http://localhost:5000")
    print("Open http://localhost:5000 in your browser for the demo.")
    app.run(host='0.0.0.0', port=5000, debug=False)
