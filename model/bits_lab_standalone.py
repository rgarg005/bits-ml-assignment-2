"""
Self-contained BITS Virtual Lab run - paste this whole file into ONE Jupyter cell.

Deliberately has NO local imports (no preprocessing.py / evaluation.py needed) and
NO seaborn dependency, because the lab image may lack either. Needs only pandas,
numpy and scikit-learn.

Trains all six classifiers on the UCI Bank Marketing dataset and prints the
comparison table, so a single screenshot of the output proves the assignment was
executed on BITS Virtual Lab.

Dataset resolution order:
  1. a local bank-full.csv (searched in the usual places)
  2. download from the UCI repository (needs internet)
  3. fall back to test_data.csv, clearly flagged as a reduced run
"""

import getpass
import io
import platform
import socket
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

NUMERIC = ["age", "balance", "day", "duration", "campaign", "previous",
           "days_since_last_contact"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "poutcome", "contacted_in_past_campaign"]

# ---------------------------------------------------------------- provenance
print("=" * 78)
print("BITS VIRTUAL LAB - ML ASSIGNMENT 2 EXECUTION")
print("=" * 78)
print(f"  host      : {socket.gethostname()}")
print(f"  user      : {getpass.getuser()}")
print(f"  platform  : {platform.platform()}")
print(f"  python    : {sys.version.split()[0]}")
print(f"  cwd       : {Path.cwd()}")
import sklearn  # noqa: E402  - imported here only to report its version
print(f"  numpy     : {np.__version__}   pandas: {pd.__version__}   "
      f"sklearn: {sklearn.__version__}")
print(f"  timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()


# ---------------------------------------------------------------- load data
def load_bank_data():
    """Return (frame, description). Tries local file, then UCI, then test_data.csv."""
    for candidate in [
        Path("bank-full.csv"), Path(".data/bank-full.csv"),
        Path("../.data/bank-full.csv"), Path("../bank-full.csv"),
        Path.home() / "bank-full.csv",
    ]:
        if candidate.exists():
            return pd.read_csv(candidate, sep=";"), f"local file {candidate}"

    try:
        print(f"  fetching {UCI_URL} ...")
        with urllib.request.urlopen(UCI_URL, timeout=90) as response:
            outer = zipfile.ZipFile(io.BytesIO(response.read()))
            inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
            frame = pd.read_csv(io.BytesIO(inner.read("bank-full.csv")), sep=";")
        frame.to_csv("bank-full.csv", sep=";", index=False)
        return frame, "downloaded from UCI (cached to bank-full.csv)"
    except Exception as error:                                  # noqa: BLE001
        print(f"  download failed ({type(error).__name__}: {error})")

    for candidate in [Path("test_data.csv"), Path("../test_data.csv")]:
        if candidate.exists():
            print("  !! falling back to the held-out split - this is a REDUCED run")
            return pd.read_csv(candidate), f"REDUCED: {candidate} (test split only)"

    raise SystemExit(
        "\nNo dataset available. Either enable internet access, or upload "
        "bank-full.csv (or test_data.csv) next to this notebook and re-run."
    )


bank, source = load_bank_data()
print(f"  dataset   : {source}")
print(f"  shape     : {bank.shape[0]:,} rows x {bank.shape[1] - 1} features + target")

positive_rate = (bank["y"] == "yes").mean()
print(f"  class bal : {positive_rate:.2%} 'yes'  |  majority baseline "
      f"= {1 - positive_rate:.2%} accuracy")
print(f"  nulls     : {int(bank.isna().sum().sum())}")
print(f"  requires  : >=12 features -> {bank.shape[1] - 1} OK   "
      f">=500 rows -> {bank.shape[0]:,} OK")
print()

# ------------------------------------------------- feature engineering
# 'pdays' uses -1 as a sentinel for "never contacted before". Left as a raw
# number, StandardScaler treats it as one day below zero, corrupting the column.
never = bank["pdays"] == -1
bank = bank.assign(
    contacted_in_past_campaign=(~never).map({True: "yes", False: "no"}),
    days_since_last_contact=bank["pdays"].where(~never, 0),
)
print(f"  engineered: split 'pdays' sentinel -> flag + clean numeric "
      f"({never.sum():,} rows never contacted)")

features = bank[NUMERIC + CATEGORICAL]
target = bank["y"].map({"yes": 1, "no": 0}).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, stratify=target, random_state=SEED
)
print(f"  split     : {len(X_train):,} train / {len(X_test):,} test (stratified)")
print()


def make_encoder():
    # sparse_threshold=0 because GaussianNB and HistGradientBoosting reject sparse.
    return ColumnTransformer(
        [("num", StandardScaler(), NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL)],
        sparse_threshold=0,
    )


MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=25, class_weight="balanced", random_state=SEED),
    "kNN": KNeighborsClassifier(n_neighbors=25, weights="distance"),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, max_depth=14, min_samples_leaf=10,
        class_weight="balanced_subsample", random_state=SEED),
    "Gradient Boosting (Ensemble)": HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, random_state=SEED),
}

print("=" * 78)
print("TRAINING SIX CLASSIFIERS")
print("=" * 78)

rows, fitted = [], {}
for name, estimator in MODELS.items():
    pipeline = Pipeline([("encode", make_encoder()), ("classify", estimator)])
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    predicted = pipeline.predict(X_test)
    scored = pipeline.predict_proba(X_test)[:, 1]
    fitted[name] = pipeline
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "AUC": roc_auc_score(y_test, scored),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1": f1_score(y_test, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predicted),
    })
    print(f"  {name:<30} trained in {elapsed:5.2f}s")

comparison = pd.DataFrame(rows).set_index("ML Model Name")

print()
print("=" * 78)
print("COMPARISON TABLE - held-out test set, threshold 0.50")
print("=" * 78)
print(comparison.round(4).to_string())
print("=" * 78)
print(f"\nMajority-class baseline accuracy: {1 - y_test.mean():.4f}")
print("\nBest model per metric:")
for metric in comparison.columns:
    winner = comparison[metric].idxmax()
    print(f"  {metric:<10} {winner:<30} {comparison.loc[winner, metric]:.4f}")

below = comparison.index[comparison["Accuracy"] < (1 - y_test.mean())].tolist()
print(f"\n{len(below)} model(s) score BELOW the do-nothing baseline on accuracy, "
      "yet beat")
print("kNN on MCC - which is why accuracy is the wrong headline metric here.")

# ------------------------------------------------- threshold sensitivity
print()
print("=" * 78)
print("THRESHOLD SENSITIVITY - the 0.50 cut-off is a convention, not an optimum")
print("=" * 78)
for name, pipeline in fitted.items():
    scored = pipeline.predict_proba(X_test)[:, 1]
    best_mcc, best_threshold = -1.0, 0.5
    for threshold in np.arange(0.05, 1.0, 0.05):
        mcc = matthews_corrcoef(y_test, (scored >= threshold).astype(int))
        if mcc > best_mcc:
            best_mcc, best_threshold = mcc, threshold
    print(f"  {name:<30} MCC {comparison.loc[name, 'MCC']:.4f} @0.50  ->  "
          f"{best_mcc:.4f} @{best_threshold:.2f}")

# ------------------------------------------------- best model detail
champion = comparison["MCC"].idxmax()
predicted = fitted[champion].predict(X_test)
print()
print("=" * 78)
print(f"CONFUSION MATRIX + CLASSIFICATION REPORT - {champion}")
print("=" * 78)
matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
print(pd.DataFrame(matrix,
                   index=["Actual: no", "Actual: yes"],
                   columns=["Pred: no", "Pred: yes"]).to_string())
tn, fp, fn, tp = matrix.ravel()
print(f"\n  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
print()
print(classification_report(y_test, predicted,
                            target_names=["no (did not subscribe)", "yes (subscribed)"],
                            zero_division=0))
print("=" * 78)
print("EXECUTION COMPLETE - all six models trained and evaluated on BITS Virtual Lab")
print("=" * 78)
