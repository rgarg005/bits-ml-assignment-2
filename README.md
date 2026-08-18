# Term Deposit Subscription Predictor

BITS Pilani WILP, M.Tech (AIML) — Machine Learning Assignment 2

| | |
|---|---|
| GitHub repository | <https://github.com/rgarg005/bits-ml-assignment-2> |
| Live Streamlit app | <https://bits-ml-assignment-2-dxnqhzykh5yl6dd9bhk4ev.streamlit.app> |

---

## a. Problem statement

A Portuguese bank runs outbound telephone campaigns to sell term deposits. Calling
every client on the list is expensive and most calls do not convert, so the bank
wants to know in advance which clients are likely to subscribe.

Given a client's demographics, financial position and contact history, predict
whether that client will subscribe to a term deposit. The target column `y` takes
the values `yes` and `no`, so this is a binary classification problem.

Only 11.7% of contacted clients subscribed. This imbalance matters when reading the
results: a model that predicts `no` for every client scores 88.3% accuracy while
being useless in practice. Accuracy is therefore reported but not relied on, and
MCC and AUC are used to compare the models.

## b. Dataset description

UCI Bank Marketing dataset (Moro, Cortez and Rita, 2014), dataset id 222,
file `bank-full.csv`: <https://archive.ics.uci.edu/dataset/222/bank+marketing>

| Property | Value |
|---|---|
| Instances | 45,211 (requirement: at least 500) |
| Features | 16 (requirement: at least 12) |
| Target | `y` — subscribed to a term deposit, yes or no |
| Class balance | 88.3% no, 11.7% yes |
| Missing values | None. Unknown entries are encoded as the category "unknown" |
| Train/test split | Stratified 75/25, `random_state=42` — 33,908 train, 11,303 test |

The 16 features are `age`, `job`, `marital`, `education`, `default`, `balance`,
`housing`, `loan`, `contact`, `day`, `month`, `duration`, `campaign`, `pdays`,
`previous` and `poutcome`.

Two preprocessing decisions are worth recording. First, `pdays` holds the number of
days since the client was last contacted but uses `-1` to mean "never contacted in a
previous campaign". Scaling that value directly would treat a never-contacted client
as a real numeric quantity, so it was split into a yes/no flag and a numeric column
that is 0 when the client was never contacted. This takes the feature count from 16
to 17, and to 43 columns after one-hot encoding. Second, numeric features are
standardised and categorical features one-hot encoded through the same
`ColumnTransformer` for all six models, so the comparison reflects the algorithms
rather than differences in preprocessing.

## c. GitHub repository link

<https://github.com/rgarg005/bits-ml-assignment-2>

```
bits-ml-assignment-2/
|-- app.py                  Streamlit application
|-- requirements.txt        Pinned dependencies
|-- README.md               This file
|-- test_data.csv           Held-out test split, 11,303 rows
`-- model/
    |-- preprocessing.py    Feature engineering and encoder
    |-- evaluation.py       The six evaluation metrics
    |-- train_models.py     Trains and saves all six models
    |-- BITS_Lab_ML_Assignment2.ipynb   Notebook run on BITS Virtual Lab
    |-- metrics.json        Comparison table values
    `-- *.joblib            Six saved scikit-learn pipelines
```

## d. Models used

All six models are scikit-learn `Pipeline` objects sharing the same
`ColumnTransformer`, trained on the same 75% split and evaluated on the same 11,303
held-out rows. Where the estimator supports it, the minority class is up-weighted
with `class_weight`; `KNeighborsClassifier` and `GaussianNB` do not expose that
parameter, which affects their results.

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight='balanced'` |
| Decision Tree | `max_depth=8`, `min_samples_leaf=25`, `class_weight='balanced'` |
| kNN | `n_neighbors=25`, `weights='distance'` |
| Naive Bayes | `GaussianNB` with default priors |
| Random Forest (Ensemble) | `n_estimators=200`, `max_depth=14`, `min_samples_leaf=10`, `class_weight='balanced_subsample'` |
| Gradient Boosting (Ensemble) | `HistGradientBoostingClassifier`, `max_iter=300`, `learning_rate=0.08` |

A sixth model is included because the assignment text refers to six models while
listing five.

### Comparison table

Held-out test set, 11,303 rows, decision threshold 0.50.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8477 | 0.9085 | 0.4218 | 0.8162 | 0.5562 | 0.5130 |
| Decision Tree | 0.7963 | 0.8965 | 0.3521 | 0.8828 | 0.5035 | 0.4713 |
| kNN | 0.8963 | 0.8942 | 0.6471 | 0.2496 | 0.3603 | 0.3586 |
| Naive Bayes | 0.8556 | 0.8052 | 0.4041 | 0.4939 | 0.4445 | 0.3649 |
| Random Forest (Ensemble) | 0.8489 | 0.9239 | 0.4268 | 0.8510 | 0.5685 | 0.5317 |
| Gradient Boosting (Ensemble) | 0.9078 | 0.9327 | 0.6489 | 0.4614 | 0.5393 | 0.4985 |

Majority-class baseline accuracy: 0.8830.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performs better than expected for a linear model, with the third-best MCC of 0.5130 and an AUC of 0.9085, and it trains in about 0.1 seconds. Its accuracy of 0.8477 is below the majority-class baseline, which is a consequence of `class_weight='balanced'` rather than a fault: the model accepts false positives in order to reach a recall of 0.8162. At a precision of 0.4218 this means roughly 2.4 calls per conversion. It is a reasonable baseline for this problem and the ensembles improve on its MCC by only about 0.02. |
| Decision Tree | Records the lowest accuracy in the table at 0.7963, which is 8.7 points below the baseline, yet its MCC of 0.4713 is higher than kNN's 0.3586 despite kNN scoring 10 points more on accuracy. This is the clearest illustration in the results that accuracy is misleading on an imbalanced dataset. The tree achieves the best recall of 0.8828 and the worst precision of 0.3521. Depth was limited to 8 with a minimum of 25 samples per leaf; without those limits it memorises the training split. Its weakness is the variance of a single tree, which is what the two ensembles address. |
| kNN | Appears to be the weakest model at a threshold of 0.50, but this is misleading. Its recall of 0.2496 means it misses 992 of the 1,322 subscribers in the test set, and its accuracy of 0.8963 is flattered by the class imbalance. The cause is that `KNeighborsClassifier` has no `class_weight` parameter, so with 11.7% positives a vote among 25 neighbours rarely reaches a 0.50 majority. Its AUC of 0.8942 is competitive, and lowering the threshold to 0.20 raises MCC from 0.3586 to 0.5148, ahead of the Decision Tree. The ranking was sound; the cut-off was wrong. |
| Naive Bayes | The weakest model, and the only one whose weakness is not explained by the threshold: its AUC of 0.8052 is well below the 0.8942 to 0.9327 range of the others, and tuning the threshold improves MCC only from 0.3649 to 0.3688. The cause is the conditional independence assumption, which this dataset violates. `pdays`, `previous` and `poutcome` all describe the same previous campaign, and `job`, `education` and `balance` are correlated, so treating them as independent counts the same evidence more than once. Its advantages are that training is effectively instantaneous and that it needs very little data. |
| Random Forest (Ensemble) | The best model at the default threshold, with the highest F1 of 0.5685 and the highest MCC of 0.5317. Bagging with `class_weight='balanced_subsample'` removes the variance of the single tree while keeping its recall: 0.8510 recall at 0.4268 precision, identifying 1,125 of 1,322 subscribers and missing 197. A larger forest of 300 trees at depth 18 reached an AUC of 0.9275 but produced a 58 MB artifact, so the smaller configuration was used instead: it gives an AUC of 0.9239 at 14 MB, trading 0.4% of ranking quality for a model four times smaller. |
| Gradient Boosting (Ensemble) | The strongest model on the threshold-independent measure, with the best AUC of 0.9327, and it also leads on accuracy at 0.9078 and precision at 0.6489. Its MCC of 0.4985 is below the Random Forest's because it was trained without class weighting, so it stays conservative and its recall falls to 0.4614. Adjusting the threshold to 0.25 raises its MCC to 0.5838, the highest figure any of the six models reaches. Boosting sequentially on residuals extracts more signal from the ordinal features such as `duration`, `balance` and `age` than the forest's parallel bagging does. |
| Overall winner for your dataset? | Gradient Boosting, with Random Forest as the better choice if the threshold cannot be tuned. Gradient Boosting has the highest AUC at 0.9327, which is independent of the threshold, and the best achievable MCC at 0.5838. The Random Forest wins the table as printed only because its class weighting effectively shifts the threshold for it. Two qualifications: the six models span only 0.9327 to 0.8052 in AUC, whereas retuning the threshold moved kNN's MCC by 0.156, so the operating point matters more than the choice of algorithm on this dataset; and `duration` accounts for 46.9% of the Random Forest's feature importance despite being the length of the call, which is not known until after the call has been made. A model intended to select who to call would have to exclude it, and doing so lowers AUC by about 0.13. |

## e. Streamlit app features

<https://bits-ml-assignment-2-dxnqhzykh5yl6dd9bhk4ev.streamlit.app>

| Required feature | Implementation |
|---|---|
| Dataset upload (CSV) | Sidebar file uploader. Accepts comma or semicolon separated files and falls back to the bundled `test_data.csv` |
| Model selection dropdown | Sidebar dropdown listing all six trained models; the test set is re-scored on selection |
| Display of evaluation metrics | All six metrics shown for the selected model, and a comparison table across all six models |
| Confusion matrix / classification report | Labelled confusion matrix, ROC curve with AUC, and a per-class classification report |

The app also provides a decision-threshold slider, since the results above show that
the 0.50 cut-off is not the best operating point for every model.
