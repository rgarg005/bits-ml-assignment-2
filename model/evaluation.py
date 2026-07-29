"""
The six evaluation metrics required by the assignment, in one place.

Shared by the training script and the Streamlit app so the numbers in the
README and the numbers rendered in the browser are produced by identical code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

METRIC_HELP = {
    "Accuracy": "Share of all clients classified correctly. Inflated here: always "
                "predicting 'no' already scores ~88% on this dataset.",
    "AUC": "Area under the ROC curve - ranking quality across every threshold, "
           "and the metric least distorted by the class imbalance.",
    "Precision": "Of the clients the model flagged as likely subscribers, the "
                 "share that actually subscribed. Governs wasted call effort.",
    "Recall": "Of the clients who actually subscribed, the share the model "
              "caught. Governs revenue left on the table.",
    "F1": "Harmonic mean of precision and recall - a single number for the "
          "trade-off between wasted calls and missed customers.",
    "MCC": "Matthews correlation coefficient. Uses all four confusion-matrix "
           "cells, so it is the honest headline number on imbalanced data. "
           "0 means no better than chance.",
}


def positive_class_scores(fitted_model, features: pd.DataFrame) -> np.ndarray:
    """Continuous score for the positive class, needed for a threshold-free AUC.

    Every estimator in this project exposes predict_proba, but decision_function
    is accepted as a fallback so the helper keeps working if the model zoo is
    extended with an SVM.
    """
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(features)[:, 1]
    if hasattr(fitted_model, "decision_function"):
        return fitted_model.decision_function(features)
    raise AttributeError(
        f"{type(fitted_model).__name__} exposes neither predict_proba nor "
        "decision_function, so an AUC score cannot be computed."
    )


def score_classifier(fitted_model, features: pd.DataFrame, truth: pd.Series) -> dict:
    """Compute all six assignment metrics for one fitted model."""
    predictions = fitted_model.predict(features)
    scores = positive_class_scores(fitted_model, features)

    return {
        "Accuracy": float(accuracy_score(truth, predictions)),
        "AUC": float(roc_auc_score(truth, scores)),
        # zero_division=0 keeps the table populated if a model degenerates into
        # predicting the majority class for every row (precision would be 0/0).
        "Precision": float(precision_score(truth, predictions, zero_division=0)),
        "Recall": float(recall_score(truth, predictions, zero_division=0)),
        "F1": float(f1_score(truth, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(truth, predictions)),
    }


def score_at_threshold(
    fitted_model,
    features: pd.DataFrame,
    truth: pd.Series,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray]:
    """Score a model at an arbitrary decision threshold.

    The default 0.5 cut-off is an arbitrary convention, not a property of the
    data. On a campaign with an 11.7% subscribe rate, the threshold is really a
    business lever: lower it to make more calls and catch more subscribers,
    raise it to spend less effort per conversion. AUC is unaffected (it
    integrates over every threshold); precision, recall, F1 and MCC all move.

    Returns (metrics, predictions) so the caller can reuse the hard labels for
    the confusion matrix without recomputing them.
    """
    scores = positive_class_scores(fitted_model, features)
    predictions = (scores >= threshold).astype(int)

    metrics = {
        "Accuracy": float(accuracy_score(truth, predictions)),
        "AUC": float(roc_auc_score(truth, scores)),
        "Precision": float(precision_score(truth, predictions, zero_division=0)),
        "Recall": float(recall_score(truth, predictions, zero_division=0)),
        "F1": float(f1_score(truth, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(truth, predictions)),
    }
    return metrics, predictions


def confusion_frame(truth: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    """Labelled 2x2 confusion matrix, readable without decoding 0/1."""
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["Actual: no", "Actual: yes"],
        columns=["Predicted: no", "Predicted: yes"],
    )
