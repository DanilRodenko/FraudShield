from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(y_true, y_score) -> dict:
    """Returns PR-AUC and ROC-AUC for predicted fraud probabilities."""
    return {
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
    }