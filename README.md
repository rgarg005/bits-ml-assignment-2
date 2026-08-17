# Term Deposit Subscription Predictor

**BITS Pilani WILP — M.Tech (AIML) — Machine Learning Assignment 2**

Six classification models trained on the UCI Bank Marketing dataset, wrapped in an
interactive Streamlit dashboard that evaluates any uploaded test CSV live.

| | |
|---|---|
| **Live app** | <https://bits-ml-assignment-2-dxnqhzykh5yl6dd9bhk4ev.streamlit.app> |
| **GitHub repo** | <https://github.com/rgarg005/bits-ml-assignment-2> |

---

## a. Problem statement

A Portuguese retail bank runs outbound telemarketing campaigns to sell **term
deposits**. Calling every client on the list is expensive and most calls fail —
only **11.7%** of contacted clients subscribe.

The task is a **binary classification** problem: given a client's demographics,
financial position, and contact history, predict whether that client will
subscribe to a term deposit (`y = yes`) so the campaign can prioritise the call
list instead of dialling indiscriminately.

The business framing determines which metric matters, and this is the central
analytical decision in this assignment:

- A **false positive** wastes one sales call — cheap.
- A **false negative** forfeits a term-deposit customer — expensive.

Because only 11.7% of rows are positive, a model that blindly predicts "no" for
everyone already scores **88.3% accuracy** while being commercially worthless.
**Accuracy is therefore a trap on this dataset**, and MCC / AUC / recall are the
metrics that actually discriminate between the six models. Several conclusions
below invert depending on which metric you read, which is the point.

---

## b. Dataset description

**UCI Bank Marketing** — Moro, S., Cortez, P. & Rita, P. (2014), *A Data-Driven
Approach to Predict the Success of Bank Telemarketing*, Decision Support Systems.
<https://archive.ics.uci.edu/dataset/222/bank+marketing> (file: `bank-full.csv`,
semicolon-delimited)

| Property | Value |
|---|---|
| Instances | **45,211** (requirement: ≥ 500) |
| Raw features | **16** (requirement: ≥ 12) |
| Features after engineering | 17 |
| Features after encoding | 43 |
| Target | `y` — did the client subscribe? (`yes` / `no`) |
| Class balance | 88.3% `no` / **11.7% `yes`** — imbalanced |
| Missing values | **None** (encoded as the category `"unknown"`, not as nulls) |
| Train / test split | Stratified 75 / 25 → 33,908 train, 11,303 test, `random_state=42` |

### Feature inventory

| Type | Features |
|---|---|
| Client demographics | `age`, `job`, `marital`, `education` |
| Financial position | `default`, `balance`, `housing`, `loan` |
| Current campaign contact | `contact`, `day`, `month`, `duration`, `campaign` |
| Previous campaign history | `pdays`, `previous`, `poutcome` |

### Preprocessing decisions

**1. `pdays` is a sentinel-encoded column, not a plain number.**
`pdays` holds the days since the client was last contacted, but uses **`-1` to mean
"never contacted in a previous campaign."** Passing that straight into
`StandardScaler` tells the model that a never-contacted client was contacted one
day *before* the epoch — it pollutes the mean and standard deviation of a real
numeric feature with a categorical flag. It was therefore split into two honest
columns:

- `contacted_in_past_campaign` — categorical `yes`/`no`
- `days_since_last_contact` — numeric, `0` when never contacted

This is why the feature count goes from 16 to 17.

**2. Uniform encoding across all six models.** Numeric features are
standardised (`StandardScaler`); categorical features are one-hot encoded with
`drop='first'` and `handle_unknown='ignore'`. Scaling is *essential* for kNN (a
pure distance metric) and for logistic-regression convergence, and *irrelevant*
to the tree models — but applying it uniformly means the comparison table
compares **algorithms**, not preprocessing choices.

**3. Class imbalance handled where the API allows it.** Logistic Regression,
Decision Tree and Random Forest were given `class_weight='balanced'`. **kNN and
Gaussian Naive Bayes have no such parameter**, and HistGradientBoosting was left
unweighted. This asymmetry turns out to explain most of the table below.

### ⚠️ Target leakage in `duration` — a measured caveat

`duration` is the length of the marketing call in seconds. It is **only known
after the call has ended**, so a model that relies on it cannot be used to decide
whom to call — the UCI documentation flags this explicitly. It is retained here
because it is part of the dataset's defined feature set, but its influence is
severe and was measured rather than assumed:

| | With `duration` | Without `duration` | Difference |
|---|---|---|---|
| Logistic Regression — AUC | 0.9085 | 0.7731 | **−0.1354** |
| Random Forest — AUC | 0.9239 | 0.7979 | **−0.1260** |
| Logistic Regression — MCC | 0.5130 | 0.2883 | **−0.2247** |
| Random Forest — MCC | 0.5317 | 0.3782 | **−0.1535** |

`duration` alone accounts for **46.9% of the Random Forest's total feature
importance** — more than the other 42 encoded features combined. Every headline
number in this report is therefore optimistic relative to what a genuinely
deployable pre-call model would achieve. A production version of this system
would drop `duration` and accept AUC ≈ 0.80.

---

## c. GitHub repository link

<https://github.com/rgarg005/bits-ml-assignment-2>

```
bits-ml-assignment-2/
├── app.py                  Streamlit dashboard (upload → select model → metrics)
├── requirements.txt        Pinned dependencies
├── README.md               This file
├── test_data.csv           Held-out 25% split, 11,303 rows, raw format
└── model/
    ├── preprocessing.py    Feature engineering + encoder, shared by train & app
    ├── evaluation.py       The six metrics, shared by train & app
    ├── train_models.py     Trains and persists all six models
    ├── ML_Assignment2.ipynb  Notebook version (run on BITS Virtual Lab)
    ├── metrics.json        Comparison-table numbers emitted by the training run
    └── *.joblib            Six fitted sklearn Pipelines
```

`preprocessing.py` and `evaluation.py` are imported by **both** `train_models.py`
and `app.py`. That is deliberate: it makes it structurally impossible for the
numbers in this README to drift from the numbers the app displays, which is the
usual source of train/serve skew.

### Reproducing

```bash
pip install -r requirements.txt

# fetch the dataset (bank-full.csv lands in .data/)
mkdir -p .data && cd .data
curl -sSLO https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
unzip -o bank+marketing.zip && unzip -o bank.zip && cd ..

python model/train_models.py --data .data/bank-full.csv   # retrain + regenerate
streamlit run app.py                                      # launch dashboard
```

---

## d. Models used

All six models are `sklearn.pipeline.Pipeline` objects sharing an identical
`ColumnTransformer`, trained on the same stratified 75% split and evaluated on the
same held-out 11,303 rows.

| # | Model | Key hyperparameters |
|---|---|---|
| 1 | Logistic Regression | `max_iter=2000`, `class_weight='balanced'` |
| 2 | Decision Tree | `max_depth=8`, `min_samples_leaf=25`, `class_weight='balanced'` |
| 3 | kNN | `n_neighbors=25`, `weights='distance'` |
| 4 | Naive Bayes | `GaussianNB` (default priors) |
| 5 | Random Forest (Ensemble) | `n_estimators=200`, `max_depth=14`, `min_samples_leaf=10`, `class_weight='balanced_subsample'` |
| 6 | Gradient Boosting (Ensemble) | `HistGradientBoostingClassifier`, `max_iter=300`, `lr=0.08` |

> Model 6 is an addition. The assignment text lists five models but refers to "all
> the 6 ML models", so a second ensemble was included to cover the discrepancy.

### Comparison table

Held-out test set (11,303 rows), decision threshold **0.50**. Best value per
column in **bold**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8477 | 0.9085 | 0.4218 | 0.8162 | 0.5562 | 0.5130 |
| Decision Tree | 0.7963 | 0.8965 | 0.3521 | **0.8828** | 0.5035 | 0.4713 |
| kNN | 0.8963 | 0.8942 | 0.6471 | 0.2496 | 0.3603 | 0.3586 |
| Naive Bayes | 0.8556 | 0.8052 | 0.4041 | 0.4939 | 0.4445 | 0.3649 |
| Random Forest (Ensemble) | 0.8489 | 0.9239 | 0.4268 | 0.8510 | **0.5685** | **0.5317** |
| Gradient Boosting (Ensemble) | **0.9078** | **0.9327** | **0.6489** | 0.4614 | 0.5393 | 0.4985 |

*Majority-class baseline accuracy: **0.8830**.*

**Read that baseline row again.** Logistic Regression (0.8477) and Decision Tree
(0.7963) both score *below* the accuracy of a model that predicts "no" for every
single client — yet both have far higher MCC than kNN, which *beats* the baseline
on accuracy. Any conclusion drawn from the accuracy column alone would be exactly
backwards.

### Confusion-matrix breakdown

Raw counts behind the table above, at threshold 0.50. Test set = 9,981 `no` +
1,322 `yes`. **Positive class convention: `y = yes` → 1**, so *recall* is the
share of genuine subscribers found and *FN* is a subscriber the campaign would
never have called.

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 8,502 | 1,479 | 243 | 1,079 |
| Decision Tree | 7,834 | 2,147 | **155** | **1,167** |
| kNN | 9,801 | **180** | 992 | 330 |
| Naive Bayes | 9,018 | 963 | 669 | 653 |
| Random Forest (Ensemble) | 8,470 | 1,511 | 197 | 1,125 |
| Gradient Boosting (Ensemble) | 9,651 | 330 | 712 | 610 |

The FN column is the commercially decisive one, and it spans **155 to 992** — a
6.4× range. kNN and Gradient Boosting look precise because they barely predict
`yes` at all: kNN raises only 510 positive flags in total and misses 992 of 1,322
subscribers. The Decision Tree's "worst accuracy in the table" buys the **lowest
false-negative count of any model**, which on a campaign where a missed customer
costs more than a wasted call is the trade you would actually want.

### Threshold sensitivity

The 0.50 cut-off is a convention, not a property of the data. Sweeping it from
0.05 to 0.95 and recording each model's best MCC changes the ranking:

| Model | MCC @ 0.50 | Best MCC | At threshold |
|---|---|---|---|
| Logistic Regression | 0.5130 | 0.5288 | 0.65 |
| Decision Tree | 0.4713 | 0.5036 | 0.65 |
| kNN | 0.3586 | **0.5148** | **0.20** |
| Naive Bayes | 0.3649 | 0.3688 | 0.70 |
| Random Forest (Ensemble) | 0.5317 | 0.5463 | 0.65 |
| Gradient Boosting (Ensemble) | 0.4985 | **0.5838** | **0.25** |

This single table reframes two of the results below, and is the reason the app
exposes a threshold slider rather than hard-coding 0.50.

---

## Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Far stronger than a linear model has any right to be here — AUC 0.9085, third-best MCC, and it trains in 0.1 s. Its accuracy (0.8477) is *below* the majority-class baseline, but that is a feature of `class_weight='balanced'`, not a defect: the model deliberately trades false positives for recall 0.8162, catching 82% of subscribers. Its precision of 0.4218 means ~2.4 calls per conversion. It is the correct baseline for this problem and the ensembles only beat it by ~0.02 MCC. |
| **Decision Tree** | The clearest illustration that accuracy misleads. It has the **worst accuracy in the table (0.7963)** — 8.7 points below the do-nothing baseline — yet its MCC (0.4713) beats kNN's (0.3586), which scores 10 points higher on accuracy. Achieves the **best recall (0.8828)** but the worst precision (0.3521), i.e. ~2.8 calls per conversion. Depth was capped at 8 with `min_samples_leaf=25`; unconstrained, it memorises the training split. Single-tree variance is its real weakness, which is precisely what the two ensembles fix. |
| **kNN** | Looks like the weakest model at threshold 0.50 and is not — this was the most instructive result in the assignment. Its recall of 0.2496 means it **misses 992 of 1,322 actual subscribers**, giving a poor MCC of 0.3586, and its 0.8963 accuracy is deceptive flattery from the imbalance. But `KNeighborsClassifier` exposes **no `class_weight` parameter**, so with 11.7% positives a 25-neighbour vote almost never reaches a 0.50 majority. Its AUC (0.8942) was competitive all along: **drop the threshold to 0.20 and MCC jumps 0.3586 → 0.5148**, overtaking the Decision Tree. The ranking was sound; the *cut-off* was wrong. |
| **Naive Bayes** | Genuinely the weakest model, and the only one whose weakness is not a threshold artifact: **worst AUC by a wide margin (0.8052 vs 0.8942–0.9327)**, and threshold tuning lifts MCC only 0.3649 → 0.3688. The cause is its conditional-independence assumption, which this dataset violates structurally — `pdays`/`previous`/`poutcome` all describe the same previous campaign, and `job`/`education`/`balance` are strongly correlated. Multiplying those as independent likelihoods double-counts the same evidence. Redeeming feature: instantaneous training and the strongest inductive bias for very small samples. |
| **Random Forest (Ensemble)** | **Best model at the default threshold** — top F1 (0.5685) and top MCC (0.5317), the two metrics that survive an 11.7% positive rate. Bagging plus `class_weight='balanced_subsample'` fixes the single tree's variance while preserving its recall: 0.8510 recall at 0.4268 precision, i.e. it catches 1,125 of 1,322 subscribers, missing only 197. Note the deployment trade recorded in `train_models.py`: a 300-tree/depth-18 forest scored AUC 0.9275 but pickled to 58 MB, risking the ~1 GB memory ceiling on Streamlit's free tier; the shipped 200-tree/depth-14 configuration gives AUC 0.9239 at 14 MB — **0.4% of ranking quality for a 4× smaller artifact**. |
| **Gradient Boosting (Ensemble)** | The strongest *learner*, disguised by the default threshold. It wins accuracy (0.9078), **AUC (0.9327)** and precision (0.6489), but its MCC at 0.50 (0.4985) sits below Random Forest's because it was trained **unweighted** — so it stays conservative and its recall collapses to 0.4614. AUC is threshold-independent and it leads there, which is the tell. Tune the cut-off and it converts: **MCC 0.5838 at threshold 0.25 — the best score any model achieves anywhere in this study.** Sequential boosting on residuals extracts more signal from the ordinal features (`duration`, `balance`, `age`) than the forest's parallel bagging. |
| **Overall winner for your dataset?** | **Gradient Boosting, on the evidence — with Random Forest as the pragmatic default.** Gradient Boosting has the best *intrinsic* discriminative power (AUC 0.9327, threshold-independent) and the best achievable decisions (MCC 0.5838 @ 0.25). Random Forest wins the table as printed only because `class_weight='balanced'` pre-shifts its effective threshold — it is the better out-of-the-box choice and the safer pick if the operating point cannot be tuned. **Two caveats keep this honest:** (1) the six models span only 0.9327 to 0.8052 AUC — a spread of **0.128** — whereas retuning the cut-off alone moved kNN's MCC by **0.156**. Threshold choice mattered more than algorithm choice on this dataset; (2) all of these numbers lean on `duration`, which is unavailable before a call is placed. Strip it out and the honest ceiling is AUC ≈ 0.80. |

---

## e. Streamlit app features

Live app: <https://bits-ml-assignment-2-dxnqhzykh5yl6dd9bhk4ev.streamlit.app>

| Required feature | Where |
|---|---|
| **Dataset upload (CSV)** | Sidebar → *1. Test data*. Sniffs comma- **or** semicolon delimiters, so the raw UCI file works unmodified. Falls back to the bundled `test_data.csv`. Validates columns and reports a readable error rather than a stack trace. |
| **Model selection dropdown** | Sidebar → *2. Model*. All six classifiers; switching re-scores the test set live. |
| **Display of evaluation metrics** | *Selected model* tab — all six metrics as cards, each with a tooltip explaining what it means **on this dataset**. |
| **Confusion matrix / classification report** | *Selected model* tab — labelled confusion-matrix heatmap, ROC curve with AUC, and a full per-class classification report. |

Beyond the requirements:

- **Decision-threshold slider** — makes the precision/recall trade-off in the
  observations above directly explorable, and demonstrates the kNN result live.
- **All six models tab** — the full comparison table computed on *your* uploaded
  data, best-per-metric highlighted, plus re-rankable bar charts by any metric.
- **Majority-class baseline** shown next to accuracy, so the 88.3% trap is
  visible rather than something the reader has to know.
- **Feature-influence panel** — coefficients for Logistic Regression, impurity
  importances for the tree models, and an honest "not available" for kNN, Naive
  Bayes and HistGradientBoosting rather than a fabricated chart.

---

## Notes

- `scikit-learn` is pinned to `1.7.2` in `requirements.txt`. Unpickling a
  `Pipeline` under a different minor release raises
  `InconsistentVersionWarning` and can fail outright — this is the most common
  cause of a broken Streamlit Cloud deployment.
- `random_state=42` throughout, so every number in this README reproduces exactly.
- `.data/` is git-ignored; the training data is fetched by the commands above
  rather than committed.
