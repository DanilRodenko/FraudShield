from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def make_logreg():
    # LogReg can't handle NaN and is sensitive to scale
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def make_tree():
    # Tree needs imputer (no NaN support); max_depth limits overfitting.
    return make_pipeline(
        SimpleImputer(strategy='median'),
        DecisionTreeClassifier(random_state=42, max_depth=8)
    )


def make_xgb():
    # XGBoost handles NaN natively -> no pipeline needed
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )


def make_lgbm():
    # LightGBM handles NaN natively; verbose=-1 silences training logs
    return LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
    )


MODELS = {
    "logreg": make_logreg,
    "tree": make_tree,
    "xgb": make_xgb,
    "lgbm": make_lgbm,
}