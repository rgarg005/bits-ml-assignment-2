"""
Trains and persists six classifiers on the UCI Bank Marketing dataset.

Run from the repository root:
    python model/train_models.py --data .data/bank-full.csv

Outputs (all written into model/ and the repo root):
    model/<slug>.joblib      one fitted Pipeline per algorithm
    model/metrics.json       the comparison-table numbers, for the README
    test_data.csv            the held-out split in RAW form, for the app
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import build_column_transformer, split_features_and_target

RANDOM_SEED = 42
TEST_FRACTION = 0.25

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parent


def build_model_zoo() -> dict[str, object]:
    """The six estimators required by the assignment, keyed by display name.

    Where an algorithm supports it, the minority class (~11.7% of rows) is
    up-weighted. Logistic Regression, Decision Tree and Random Forest all
    expose class_weight; kNN and Gaussian Naive Bayes do not, which is itself
    a finding worth reporting in the observations table.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            # An unconstrained tree memorises the training split and collapses
            # on held-out data; these limits are the cheapest effective guard.
            max_depth=8,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "kNN": KNeighborsClassifier(
            n_neighbors=25,
            weights="distance",
            n_jobs=-1,
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            # Deliberately pruned for deployment. A 300-tree / depth-18 forest
            # scored AUC 0.9275 but pickled to 58 MB, which risks the ~1 GB
            # memory ceiling on Streamlit Community Cloud once all six models
            # are held in memory. These settings give AUC 0.9239 at 14 MB -
            # 0.4% of ranking quality traded for a 4x smaller artefact.
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "Gradient Boosting (Ensemble)": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_SEED,
        ),
    }


def slugify(display_name: str) -> str:
    """'Random Forest (Ensemble)' -> 'random_forest_ensemble'."""
    cleaned = display_name.lower().replace("(", "").replace(")", "")
    return "_".join(cleaned.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(REPO_ROOT / ".data" / "bank-full.csv"),
        help="Path to the semicolon-delimited UCI bank-full.csv",
    )
    args = parser.parse_args()

    # Local import so the module works whether it is run as a script or a
    # notebook cell in the BITS Virtual Lab.
    from evaluation import score_classifier

    bank_df = pd.read_csv(args.data, sep=";")
    print(f"Loaded {bank_df.shape[0]:,} rows x {bank_df.shape[1] - 1} raw features")

    features, target = split_features_and_target(bank_df)

    # Stratified split preserves the 11.7% positive rate in both halves, which
    # is what makes the held-out AUC and MCC comparable to the training run.
    feat_train, feat_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=TEST_FRACTION,
        stratify=target,
        random_state=RANDOM_SEED,
    )
    print(
        f"Train: {len(feat_train):,} rows ({target_train.mean():.1%} positive) | "
        f"Test: {len(feat_test):,} rows ({target_test.mean():.1%} positive)"
    )

    # Persist the held-out split in RAW form (original columns, yes/no label)
    # so that the Streamlit upload path exercises the same engineering code the
    # training run used, rather than a pre-cooked matrix.
    raw_test_split = bank_df.loc[feat_test.index]
    test_csv_path = REPO_ROOT / "test_data.csv"
    raw_test_split.to_csv(test_csv_path, index=False)
    print(f"Wrote held-out test set -> {test_csv_path} ({len(raw_test_split):,} rows)")

    results = []
    for display_name, estimator in build_model_zoo().items():
        pipeline = Pipeline(
            steps=[
                ("encode", build_column_transformer()),
                ("classify", estimator),
            ]
        )

        started = time.perf_counter()
        pipeline.fit(feat_train, target_train)
        fit_seconds = time.perf_counter() - started

        scores = score_classifier(pipeline, feat_test, target_test)
        scores["Model"] = display_name
        scores["fit_seconds"] = round(fit_seconds, 2)
        results.append(scores)

        artefact_path = MODEL_DIR / f"{slugify(display_name)}.joblib"
        joblib.dump(pipeline, artefact_path, compress=3)
        size_mb = artefact_path.stat().st_size / 1024**2

        print(
            f"{display_name:<30} acc={scores['Accuracy']:.4f} auc={scores['AUC']:.4f} "
            f"f1={scores['F1']:.4f} mcc={scores['MCC']:.4f} "
            f"({fit_seconds:5.1f}s, {size_mb:.1f} MB)"
        )

    comparison = pd.DataFrame(results).set_index("Model")
    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

    print("\n=== Comparison table (held-out test split) ===")
    print(comparison[metric_columns].round(4).to_string())

    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": Path(args.data).name,
                "n_rows": int(bank_df.shape[0]),
                "n_raw_features": int(bank_df.shape[1] - 1),
                "n_encoded_features": int(
                    pipeline.named_steps["encode"].transform(feat_test.head(1)).shape[1]
                ),
                "test_fraction": TEST_FRACTION,
                "random_seed": RANDOM_SEED,
                "positive_rate": round(float(target.mean()), 4),
                "results": comparison.reset_index().to_dict(orient="records"),
            },
            indent=2,
        )
    )
    print(f"\nWrote metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
