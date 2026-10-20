import pandas as pd
import xgboost as xgb


def shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature SHAP contributions for each row (in log-odds), plus `base_value`.
    Row sum = model's log-odds for that row -> contributions fully explain the score."""
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(contribs, columns=[*X.columns, "base_value"], index=X.index)


def top_reasons(model, X_row: pd.DataFrame, n: int = 3) -> list[dict]:
    """Top-n features pushing ONE transaction towards fraud (largest positive SHAP)."""
    contrib = shap_values(model, X_row).iloc[0].drop("base_value")
    top = contrib.sort_values(ascending=False).head(n)
    return [
        {
            "feature": feature,
            "value": float(X_row.iloc[0][feature]),   # the feature's actual value
            "impact": round(float(impact), 3),        # how much it pushed towards fraud
        }
        for feature, impact in top.items()
    ]