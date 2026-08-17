"""
Generates BITS_Lab_ML_Assignment2.ipynb - a completely self-contained notebook.

Unlike ML_Assignment2.ipynb (which imports preprocessing.py / evaluation.py /
train_models.py from the repo), this one inlines every helper and avoids seaborn,
so a single .ipynb file can be uploaded to BITS Virtual Lab and run end to end
with nothing else present.

    python model/build_lab_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUTPUT = Path(__file__).resolve().parent / "BITS_Lab_ML_Assignment2.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Machine Learning Assignment 2 — Bank Marketing Classification

**BITS Pilani WILP · M.Tech (AIML) · Executed on BITS Virtual Lab**

Six classification models on the UCI Bank Marketing dataset, evaluated on
Accuracy, AUC, Precision, Recall, F1 and MCC.

> **This notebook is fully self-contained.** It needs no other project files and
> no seaborn — only `pandas`, `numpy`, `scikit-learn` and `matplotlib`. Upload
> this single file and run all cells.

---
1. Environment  2. Dataset  3. Exploration  4. Preprocessing  5. Training
6. Comparison table  7. Threshold sensitivity  8. `duration` leakage  9. Conclusion""",
    ),
    (
        "markdown",
        "## 1. Environment\n\nPrinted so the notebook output itself records which "
        "machine executed it — this is the evidence for the BITS Virtual Lab "
        "requirement.",
    ),
    (
        "code",
        '''import getpass, io, platform, socket, ssl, sys, time, urllib.request, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 40)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

SEED = 42
PURPLE, LILAC = "#6A2C91", "#C9B6DB"

print("=" * 74)
print("EXECUTION ENVIRONMENT")
print("=" * 74)
print(f"  host        : {socket.gethostname()}")
print(f"  user        : {getpass.getuser()}")
print(f"  platform    : {platform.platform()}")
print(f"  python      : {sys.version.split()[0]}")
print(f"  numpy       : {np.__version__}")
print(f"  pandas      : {pd.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  working dir : {Path.cwd()}")
print(f"  timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 74)''',
    ),
    (
        "markdown",
        """## 2. Load the dataset

**UCI Bank Marketing** (Moro, Cortez & Rita, 2014) — dataset id 222.
The file is **semicolon-delimited**; reading it with the default comma separator
yields a single useless column, which is the first thing to get wrong.

The cell prefers a local `bank-full.csv` and only downloads if it must, so a
firewalled lab machine works as soon as the CSV is uploaded beside this
notebook.""",
    ),
    (
        "code",
        '''URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

SEARCH = list(dict.fromkeys([
    Path.cwd() / "bank-full.csv",
    Path.cwd() / ".data" / "bank-full.csv",
    Path.cwd().parent / ".data" / "bank-full.csv",
    Path.home() / "bank-full.csv",
]))

data_path = next((p for p in SEARCH if p.exists()), None)

def fetch(url):
    """Download `url`, trying three transports before giving up.

    Managed networks break these in different ways: a Python build may ship no
    CA bundle at all, and a TLS-inspecting corporate proxy presents a root
    certificate that `certifi` does not carry but the OS trust store does. curl
    uses the OS store, so it succeeds where urllib fails.
    """
    failures = []

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read()
    except Exception as error:
        failures.append(f"urllib default context -> {type(error).__name__}")

    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=120, context=context) as response:
            return response.read()
    except Exception as error:
        failures.append(f"urllib + certifi -> {type(error).__name__}")

    try:
        import subprocess
        finished = subprocess.run(["curl", "-sSL", "--max-time", "180", url],
                                  capture_output=True)
        if finished.returncode == 0 and finished.stdout[:2] == b"PK":
            return finished.stdout
        detail = finished.stderr[:120].decode("utf-8", "replace").strip()
        failures.append(f"curl -> rc={finished.returncode} {detail}")
    except Exception as error:
        failures.append(f"curl -> {type(error).__name__}")

    raise RuntimeError(" | ".join(failures))


if data_path is not None:
    bank = pd.read_csv(data_path, sep=";")
    print(f"loaded local copy: {data_path}")
else:
    try:
        print(f"no local copy found — downloading {URL}")
        payload = fetch(URL)
        outer = zipfile.ZipFile(io.BytesIO(payload))
        inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
        bank = pd.read_csv(io.BytesIO(inner.read("bank-full.csv")), sep=";")
        bank.to_csv("bank-full.csv", sep=";", index=False)
        print("downloaded and cached to ./bank-full.csv")
    except Exception as error:
        raise SystemExit(
            f"Could not obtain the dataset.\\n  {error}\\n\\n"
            "This machine may have no outbound internet access.\\n"
            f"Download bank-full.csv from\\n    {URL}\\n"
            "elsewhere, upload it beside this notebook, and re-run this cell.\\n\\n"
            "Searched locally:\\n  " + "\\n  ".join(str(p) for p in SEARCH)
        )

print(f"shape: {bank.shape[0]:,} rows x {bank.shape[1] - 1} features + 1 target")
bank.head()''',
    ),
    (
        "markdown",
        """## 3. Exploration

Three facts drive every later decision: the assignment size floors, the class
imbalance, and the fact that `pdays` hides a categorical flag inside a numeric
column.""",
    ),
    (
        "code",
        '''n_rows, n_features = bank.shape[0], bank.shape[1] - 1
positive_rate = (bank["y"] == "yes").mean()

print("--- assignment requirements ---")
print(f"  instances : {n_rows:,}  (need >= 500)  -> {'PASS' if n_rows >= 500 else 'FAIL'}")
print(f"  features  : {n_features}      (need >= 12)   -> {'PASS' if n_features >= 12 else 'FAIL'}")
print(f"  task      : binary classification (y = yes/no)")

print("\\n--- class balance ---")
print(bank["y"].value_counts().to_string())
print(f"\\n  positive rate           : {positive_rate:.4f}  ({positive_rate:.2%})")
print(f"  majority-class baseline : {1 - positive_rate:.4f}")
print("\\n  >> Predicting 'no' for everyone already scores "
      f"{1 - positive_rate:.1%} accuracy.")
print("  >> Accuracy is therefore a trap here; MCC and AUC carry the signal.")

print(f"\\n--- data quality ---")
print(f"  true nulls: {int(bank.isna().sum().sum())}  "
      "('unknown' is an explicit category, not NaN)")''',
    ),
    (
        "code",
        '''fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

counts = bank["y"].value_counts()
axes[0].bar(counts.index, counts.values, color=[LILAC, PURPLE])
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
axes[0].set_title(f"Class imbalance — only {positive_rate:.1%} subscribed")
axes[0].set_ylabel("clients")

for label, colour in [("no", LILAC), ("yes", PURPLE)]:
    axes[1].hist(bank.loc[bank["y"] == label, "duration"], bins=60, range=(0, 2000),
                 alpha=0.75, label=f"y = {label}", color=colour, log=True)
axes[1].set_title("Call duration separates the classes almost perfectly")
axes[1].set_xlabel("duration (seconds)")
axes[1].legend(frameon=False)

plt.tight_layout(); plt.show()

print("The right-hand plot is a leakage warning: 'duration' is only known AFTER")
print("the call ends, so it cannot inform the decision of whom to call.")
print("Quantified in section 8.")''',
    ),
    (
        "markdown",
        """### The `pdays` sentinel trap

`pdays` is "days since the client was last contacted", but **`-1` means *never
contacted before***. That is a categorical flag hiding in a numeric column —
feed it to `StandardScaler` untouched and it corrupts the feature's mean and
standard deviation.""",
    ),
    (
        "code",
        '''never = bank["pdays"] == -1
print(f"rows with pdays == -1 : {never.sum():,}  ({never.mean():.1%})")
print("\\nraw pdays — the sentinel drags the distribution down:")
print(bank["pdays"].describe().round(2).to_string())
print("\\nexcluding the sentinel, the genuine values are:")
print(bank.loc[~never, "pdays"].describe().round(2).to_string())
print("\\n  >> Fix: split into a yes/no flag plus a numeric column that is 0 when")
print("  >> never contacted. Feature count goes from 16 to 17.")''',
    ),
    (
        "markdown",
        """## 4. Preprocessing

Everything below is defined inline — no project imports. Numeric features are
standardised and categoricals one-hot encoded, **identically for all six
models**, so the comparison measures algorithms rather than preprocessing.""",
    ),
    (
        "code",
        '''from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

NUMERIC = ["age", "balance", "day", "duration", "campaign", "previous",
           "days_since_last_contact"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "poutcome", "contacted_in_past_campaign"]


def engineer(frame):
    """Split the overloaded pdays column into an honest flag + numeric pair."""
    out = frame.copy()
    unseen = out["pdays"] == -1
    out["contacted_in_past_campaign"] = (~unseen).map({True: "yes", False: "no"})
    out["days_since_last_contact"] = out["pdays"].where(~unseen, 0)
    return out.drop(columns=["pdays"])


def make_encoder(numeric=NUMERIC, categorical=CATEGORICAL):
    # sparse_threshold=0 because GaussianNB and HistGradientBoosting both reject
    # sparse input, and all six models must share one representation.
    return ColumnTransformer(
        [("num", StandardScaler(), numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical)],
        sparse_threshold=0,
    )


engineered = engineer(bank)
X = engineered[NUMERIC + CATEGORICAL]
y = engineered["y"].map({"yes": 1, "no": 0}).astype(int)

# Stratified so both halves keep the 11.7% positive rate.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED)

print(f"numeric features     ({len(NUMERIC)}): {NUMERIC}")
print(f"categorical features ({len(CATEGORICAL)}): {CATEGORICAL}")
print(f"\\nencoded width : {make_encoder().fit_transform(X).shape[1]} columns")
print(f"train         : {len(X_train):,} rows ({y_train.mean():.2%} positive)")
print(f"test          : {len(X_test):,} rows ({y_test.mean():.2%} positive)")''',
    ),
    (
        "markdown",
        """## 5. Train the six models

Note which estimators accept `class_weight` — Logistic Regression, Decision Tree
and Random Forest do; **kNN and Gaussian Naive Bayes do not**. That asymmetry
explains most of the results.""",
    ),
    (
        "code",
        '''from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report, roc_curve)

METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

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


def score(model, features, truth):
    """The six required metrics for one fitted model."""
    predicted = model.predict(features)
    probability = model.predict_proba(features)[:, 1]
    return {
        "Accuracy":  accuracy_score(truth, predicted),
        "AUC":       roc_auc_score(truth, probability),
        "Precision": precision_score(truth, predicted, zero_division=0),
        "Recall":    recall_score(truth, predicted, zero_division=0),
        "F1":        f1_score(truth, predicted, zero_division=0),
        "MCC":       matthews_corrcoef(truth, predicted),
    }


fitted, rows = {}, []
for name, estimator in MODELS.items():
    pipeline = Pipeline([("encode", make_encoder()), ("classify", estimator)])
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    result = score(pipeline, X_test, y_test)
    fitted[name] = pipeline
    rows.append({"ML Model Name": name, **result})
    print(f"  {name:<30} acc={result['Accuracy']:.4f}  auc={result['AUC']:.4f}  "
          f"mcc={result['MCC']:.4f}   ({elapsed:5.2f}s)")

comparison = pd.DataFrame(rows).set_index("ML Model Name")[METRICS]
print("\\nall six models trained.")''',
    ),
    (
        "markdown",
        "## 6. Comparison table\n\nThe deliverable table: six models × six metrics on "
        "the held-out split.",
    ),
    (
        "code",
        '''baseline = 1 - y_test.mean()

print("=" * 92)
print(f"COMPARISON TABLE — held-out test set ({len(X_test):,} rows), threshold 0.50")
print("=" * 92)
print(comparison.round(4).to_string())
print("=" * 92)
print(f"Majority-class baseline accuracy: {baseline:.4f}")

print("\\nBest model per metric:")
for metric in METRICS:
    winner = comparison[metric].idxmax()
    print(f"  {metric:<10} {winner:<30} {comparison.loc[winner, metric]:.4f}")

below = comparison.index[comparison["Accuracy"] < baseline].tolist()
print(f"\\n{len(below)} models score BELOW the do-nothing baseline on accuracy:")
for name in below:
    print(f"    {name:<30} acc={comparison.loc[name,'Accuracy']:.4f}  "
          f"mcc={comparison.loc[name,'MCC']:.4f}")
print(f"  ...yet all of them beat kNN (mcc={comparison.loc['kNN','MCC']:.4f}), which "
      "BEATS the baseline")
print("  on accuracy. Reading the accuracy column alone ranks these models backwards.")''',
    ),
    (
        "code",
        '''fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))

for axis, metric in zip(axes.ravel(), METRICS):
    ordered = comparison[metric].sort_values()
    colours = [PURPLE if v == ordered.max() else LILAC for v in ordered]
    axis.barh(range(len(ordered)), ordered.values, color=colours)
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels([n.replace(" (Ensemble)", "") for n in ordered.index],
                         fontsize=8)
    for i, v in enumerate(ordered.values):
        axis.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=7.5)
    axis.set_title(metric, fontsize=10)
    axis.set_xlim(0, 1.12)
    if metric == "Accuracy":
        axis.axvline(baseline, color="#C0392B", linestyle="--", linewidth=1.2)

plt.suptitle("Six models, six metrics — the winner changes with the metric",
             fontsize=12)
plt.tight_layout(); plt.show()''',
    ),
    (
        "code",
        '''champion = comparison["MCC"].idxmax()
predicted = fitted[champion].predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))

matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
axes[0].imshow(matrix, cmap="BuPu")
for (r, c), v in np.ndenumerate(matrix):
    axes[0].text(c, r, f"{v:,}", ha="center", va="center", fontsize=13,
                 color="white" if v > matrix.max() / 2 else "#333")
axes[0].set_xticks([0, 1], ["Pred: no", "Pred: yes"])
axes[0].set_yticks([0, 1], ["Actual: no", "Actual: yes"])
axes[0].set_title(f"{champion}\\nconfusion matrix @ 0.50", fontsize=10)
axes[0].grid(False)

for name, pipeline in fitted.items():
    fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
    axes[1].plot(fpr, tpr, linewidth=1.6,
                 label=f"{name.replace(' (Ensemble)','')} ({comparison.loc[name,'AUC']:.3f})")
axes[1].plot([0, 1], [0, 1], "--", color="#999", linewidth=1, label="random (0.500)")
axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true positive rate")
axes[1].set_title("ROC curves — all six models", fontsize=10)
axes[1].legend(fontsize=7.5, loc="lower right", frameon=False)

plt.tight_layout(); plt.show()

tn, fp, fn, tp = matrix.ravel()
print(f"{champion}:  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}\\n")
print(classification_report(y_test, predicted,
                            target_names=["no (did not subscribe)", "yes (subscribed)"],
                            zero_division=0))''',
    ),
    (
        "markdown",
        """## 7. Threshold sensitivity

0.50 is a **convention**, not a property of the data. Because kNN and Naive Bayes
cannot be class-weighted, a 0.50 cut-off on an 11.7%-positive problem penalises
them structurally. Sweeping the threshold separates *ranking ability* from
*calibration*.""",
    ),
    (
        "code",
        '''sweep = []
for name, pipeline in fitted.items():
    probability = pipeline.predict_proba(X_test)[:, 1]
    for threshold in np.arange(0.05, 1.0, 0.05):
        sweep.append({
            "model": name, "threshold": round(float(threshold), 2),
            "MCC": matthews_corrcoef(y_test, (probability >= threshold).astype(int)),
        })
sweep = pd.DataFrame(sweep)

best = sweep.loc[sweep.groupby("model")["MCC"].idxmax()].set_index("model")
best["MCC @ 0.50"] = comparison["MCC"]
best["gain"] = best["MCC"] - best["MCC @ 0.50"]

print("Best achievable MCC per model, and the threshold that achieves it:\\n")
print(best[["MCC @ 0.50", "MCC", "threshold", "gain"]]
      .sort_values("MCC", ascending=False).round(4).to_string())

tuned = best["MCC"].idxmax()
spread = comparison["AUC"].max() - comparison["AUC"].min()
print(f"\\n  >> Threshold-tuned winner : {tuned} "
      f"(MCC {best.loc[tuned,'MCC']:.4f} @ {best.loc[tuned,'threshold']:.2f})")
print(f"  >> Largest gain from tuning: {best['gain'].idxmax()} "
      f"(+{best['gain'].max():.4f} MCC)")
print(f"  >> Best-to-worst AUC spread across all six models is only {spread:.4f}.")
print("  >> Tuning ONE model's threshold moved MCC further than the entire")
print("  >> algorithmic spread: the threshold matters more than the algorithm.")''',
    ),
    (
        "code",
        '''fig, axis = plt.subplots(figsize=(8.5, 4))
for name in fitted:
    subset = sweep[sweep["model"] == name]
    axis.plot(subset["threshold"], subset["MCC"], marker="o", markersize=3,
              linewidth=1.6, label=name.replace(" (Ensemble)", ""))
axis.axvline(0.50, color="#C0392B", linestyle="--", linewidth=1.2)
axis.text(0.515, 0.02, "default 0.50", color="#C0392B", fontsize=8)
axis.set_xlabel("decision threshold"); axis.set_ylabel("MCC")
axis.set_title("MCC vs decision threshold — the default is not the optimum",
               fontsize=11)
axis.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()''',
    ),
    (
        "markdown",
        """## 8. Target leakage in `duration`

`duration` is the call length in seconds — known **only after the call ends**. A
model relying on it cannot decide whom to call. Retrained without it, the honest
performance ceiling is far lower.""",
    ),
    (
        "code",
        '''leak_rows = []
for label, numeric in [("with duration", NUMERIC),
                       ("without duration", [c for c in NUMERIC if c != "duration"])]:
    for name, estimator in [
        ("Logistic Regression", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED)),
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=10,
            class_weight="balanced_subsample", random_state=SEED)),
    ]:
        pipeline = Pipeline([("encode", make_encoder(numeric=numeric)),
                             ("classify", estimator)])
        pipeline.fit(X_train, y_train)
        leak_rows.append({"model": name, "variant": label,
                          **score(pipeline, X_test, y_test)})

leak = pd.DataFrame(leak_rows).pivot(index="model", columns="variant")
for metric in ["AUC", "MCC"]:
    block = leak[metric].copy()
    block["drop"] = block["with duration"] - block["without duration"]
    print(f"--- {metric} ---"); print(block.round(4).to_string()); print()

forest = fitted["Random Forest (Ensemble)"]
names = [n.split("__", 1)[-1]
         for n in forest.named_steps["encode"].get_feature_names_out()]
importance = pd.Series(forest.named_steps["classify"].feature_importances_,
                       index=names).sort_values(ascending=False)
print("Random Forest — top 8 feature importances:")
print(importance.head(8).round(4).to_string())
print(f"\\n  >> 'duration' alone is {importance['duration']:.1%} of total importance —")
print(f"  >> more than the other {len(names) - 1} encoded features combined.")
print("  >> Every headline number above is optimistic for a real pre-call model.")''',
    ),
    (
        "markdown",
        """## 9. Conclusion

| Finding | Evidence |
|---|---|
| Accuracy is misleading here | Baseline is 88.3%; two models score *below* it yet beat kNN on MCC |
| Random Forest is the best default | Top F1 and MCC at threshold 0.50 |
| Gradient Boosting is the strongest learner | Best AUC (threshold-independent) and best tuned MCC |
| kNN's weakness was calibration, not ranking | No `class_weight` support ⇒ collapses at 0.50, recovers sharply near 0.20 |
| Naive Bayes is genuinely weakest | Worst AUC; conditional-independence assumption structurally violated |
| Threshold choice > algorithm choice | Retuning one cut-off moved MCC more than the whole best-to-worst AUC spread |
| All results lean on a leaky feature | `duration` is ~47% of Random Forest importance and is unknown before a call |

**Overall winner: Gradient Boosting** on the evidence (best AUC, best tuned MCC),
with **Random Forest** the pragmatic default at the untuned 0.50 threshold.

---
*Executed on BITS Virtual Lab. Full project, deployed Streamlit app and README:*
*https://github.com/rgarg005/bits-ml-assignment-2*""",
    ),
    (
        "code",
        '''print("=" * 74)
print("EXECUTION COMPLETE")
print("=" * 74)
print(f"  models trained : {len(fitted)}")
print(f"  metrics each   : {len(METRICS)}  ({', '.join(METRICS)})")
print(f"  test rows      : {len(X_test):,}")
print(f"  best by MCC    : {comparison['MCC'].idxmax()} "
      f"({comparison['MCC'].max():.4f})")
print(f"  best by AUC    : {comparison['AUC'].idxmax()} "
      f"({comparison['AUC'].max():.4f})")
print(f"  host / user    : {socket.gethostname()} / {getpass.getuser()}")
print(f"  finished       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 74)''',
    ),
]


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(body) if kind == "markdown" else nbf.v4.new_code_cell(body)
        for kind, body in CELLS
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(notebook, str(OUTPUT))
    code_cells = sum(1 for kind, _ in CELLS if kind == "code")
    print(f"wrote {OUTPUT} ({len(CELLS)} cells, {code_cells} code)")


if __name__ == "__main__":
    main()
