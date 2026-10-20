import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, precision_score, recall_score


def evaluate(y_true, y_score) -> dict:
    """Returns PR-AUC and ROC-AUC for predicted fraud probabilities."""
    return {
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
    }


def threshold_metrics(y_true, y_score, threshold: float) -> dict:
    """Confusion-matrix numbers + precision/recall at a given threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)

    # labels=[0, 1] keeps the matrix 2x2 even if one class is missing
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # int()/float(): plain Python types, so the dict can be saved to JSON
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }