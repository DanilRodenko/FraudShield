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
    # XGBoost handles NaN natively. Params chosen on validation (see notebook 02):
    # shallower trees + smaller steps -> smallest train/val gap of all models
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        eval_metric="logloss",
        random_state=42,
    )


def make_lgbm():
    # LightGBM is unstable on this imbalance with defaults (val PR-AUC 0.64);
    # regularised params below bring it to 0.80
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=100,
        random_state=42,
        verbose=-1,
    )


MODELS = {
    "logreg": make_logreg,
    "tree": make_tree,
    "xgb": make_xgb,
    "lgbm": make_lgbm,
}