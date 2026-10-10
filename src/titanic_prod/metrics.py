"""One place for classification metrics — train, evaluate, and compare share it.

Students: same test split + same formulas, so the MLflow table is a fair race.
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    # Core scores for a binary survived/died problem, rounded to 2 decimals
    # so MLflow tables and classroom leaderboards stay readable.
    out: dict = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 2),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 2),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 2),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 2),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True),
        "n_test": len(y_true),
    }
    # ROC AUC needs probabilities; None when the model has no predict_proba.
    # ValueError covers the single-class edge case (e.g. tiny toy splits).
    try:
        out["roc_auc"] = (round(float(roc_auc_score(y_true, y_proba)), 2)
                          if y_proba is not None else None)
    except ValueError:
        out["roc_auc"] = None
    return out
