import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from app.explain import shap_values, top_reasons


def _toy_model():
    # Synthetic rule: fraud = high amount at night -> `amt` and `is_night` must drive the score
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "amt": rng.random(2000) * 100,
        "is_night": rng.integers(0, 2, 2000),
        "noise": rng.random(2000),
    })
    y = ((X["amt"] > 80) & (X["is_night"] == 1)).astype(int)
    model = XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss").fit(X, y)
    return model, X, y


def test_contributions_sum_to_model_score():
    # SHAP additivity: base + contributions = log-odds -> sigmoid = predict_proba
    model, X, _ = _toy_model()
    log_odds = shap_values(model, X.head(20)).sum(axis=1)
    proba = 1 / (1 + np.exp(-log_odds))
    np.testing.assert_allclose(proba, model.predict_proba(X.head(20))[:, 1], rtol=1e-4)


def test_top_reason_for_fraud_is_not_noise():
    model, X, y = _toy_model()
    fraud_row = X[y == 1].head(1)
    reasons = top_reasons(model, fraud_row, n=1)
    assert reasons[0]["feature"] in {"amt", "is_night"}