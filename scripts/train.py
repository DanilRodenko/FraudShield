import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from app.data import load_data, split_xy
from app.features import FEATURE_COLS, build_features
from app.metrics import evaluate
from app.models import make_xgb

MODELS_DIR = Path("models")


def main():
    df = build_features(load_data())
    splits = split_xy(df)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # Model is already chosen on val -> use train + val together for the final fit
    X_fit = pd.concat([X_train, X_val])
    y_fit = pd.concat([y_train, y_val])

    model = make_xgb().fit(X_fit, y_fit)

    # Test is used ONCE, here, for the final reported numbers
    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_scores)

    print("test:", test_metrics)

    # --- Save artifacts ------------------------------------------------------
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")

    report = {
        "model": "xgboost",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": FEATURE_COLS,
        "n_fit_rows": len(X_fit),
        "test_rows": len(X_test),
        "test_fraud_rate": round(float(y_test.mean()), 4),
        "test_metrics": test_metrics,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"saved model + metrics to {MODELS_DIR}/")


if __name__ == "__main__":
    main()