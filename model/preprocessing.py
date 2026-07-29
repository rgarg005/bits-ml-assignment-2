"""
Shared feature engineering for the Bank Marketing term-deposit classifier.

This module is imported by BOTH the training script and the Streamlit app so
that a CSV uploaded at inference time is transformed in exactly the same way as
the training data. Keeping it here (rather than inside a pickled
FunctionTransformer) means the saved .joblib files contain nothing but plain
scikit-learn estimators, which is what makes them safe to unpickle on a
different machine / Python build.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "y"

# 'pdays' uses -1 as a sentinel for "this client was never contacted in a
# previous campaign". Left as-is, StandardScaler treats -1 as a real number one
# unit below zero days, which silently corrupts the numeric distribution.
PDAYS_NEVER_CONTACTED = -1

NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "previous",
    "days_since_last_contact",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "contacted_in_past_campaign",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw bank-marketing frame into the feature matrix the models expect.

    Splits the overloaded 'pdays' column into an honest pair:
      * contacted_in_past_campaign - categorical yes/no flag
      * days_since_last_contact    - numeric, 0 when never contacted

    Returns a new frame; the input is not mutated.
    """
    frame = raw.copy()

    if "pdays" not in frame.columns:
        raise ValueError(
            "Expected a 'pdays' column in the uploaded data. "
            "Please upload a CSV with the original UCI Bank Marketing columns."
        )

    never_contacted = frame["pdays"] == PDAYS_NEVER_CONTACTED
    frame["contacted_in_past_campaign"] = (~never_contacted).map({True: "yes", False: "no"})
    frame["days_since_last_contact"] = frame["pdays"].where(~never_contacted, 0)
    frame = frame.drop(columns=["pdays"])

    missing = [column for column in MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Uploaded data is missing required column(s): {missing}")

    return frame


def split_features_and_target(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Engineer features and peel off the binary target as 0/1 integers."""
    if TARGET_COLUMN not in raw.columns:
        raise ValueError(
            f"Uploaded data must contain the label column '{TARGET_COLUMN}' "
            "so that evaluation metrics can be computed."
        )

    engineered = engineer_features(raw)
    target = engineered[TARGET_COLUMN].map({"yes": 1, "no": 0})

    if target.isna().any():
        bad = sorted(set(engineered.loc[target.isna(), TARGET_COLUMN].astype(str)))
        raise ValueError(f"Column '{TARGET_COLUMN}' must be 'yes'/'no'; found {bad}")

    return engineered[MODEL_FEATURES], target.astype(int)


def build_column_transformer() -> ColumnTransformer:
    """Standardise the numeric block, one-hot the categorical block.

    Scaling matters unevenly across the model zoo: it is essential for kNN
    (a pure distance metric) and for Logistic Regression convergence, and
    irrelevant to the tree-based models. Applying it uniformly keeps every
    model on an identical feature representation so the comparison table is
    actually comparing algorithms rather than preprocessing choices.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                # handle_unknown='ignore' so a category absent from the training
                # split (e.g. a rare job title) does not crash inference.
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        # One-hot encoding returns a sparse matrix by default, but GaussianNB and
        # HistGradientBoostingClassifier both reject sparse input. Forcing dense
        # output here keeps a single shared transformer valid for all six models;
        # at ~40 encoded columns the memory cost is negligible.
        sparse_threshold=0,
    )
