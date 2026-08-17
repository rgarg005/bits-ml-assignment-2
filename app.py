"""
Term Deposit Subscription Predictor - interactive model comparison dashboard.

BITS WILP M.Tech (AIML) - Machine Learning Assignment 2.

Six classifiers trained on the UCI Bank Marketing dataset are loaded from
model/*.joblib and evaluated live against whatever labelled test CSV the user
uploads. Run locally with:

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, roc_curve

REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "model"

# The model package is imported by path rather than installed, so that the repo
# stays runnable as a plain checkout both locally and on Streamlit Cloud.
sys.path.insert(0, str(MODEL_DIR))

from evaluation import (  # noqa: E402  (import must follow the sys.path edit)
    METRIC_HELP,
    METRIC_ORDER,
    confusion_frame,
    positive_class_scores,
    score_at_threshold,
)
from preprocessing import split_features_and_target  # noqa: E402

# Display name -> artefact filename. Order controls the dropdown order.
MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
    "Gradient Boosting (Ensemble)": "gradient_boosting_ensemble.joblib",
}

BUNDLED_TEST_CSV = REPO_ROOT / "test_data.csv"

st.set_page_config(
    page_title="Term Deposit Predictor - Model Comparison",
    page_icon="📞",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading trained models...")
def load_model(display_name: str):
    """Load one fitted pipeline. cache_resource keeps it out of the session state
    pickle and shares a single copy across all browser sessions - important on
    the free tier, where the forest is the bulk of the memory budget."""
    artefact = MODEL_DIR / MODEL_REGISTRY[display_name]
    if not artefact.exists():
        st.error(
            f"Missing model artefact `{artefact.relative_to(REPO_ROOT)}`. "
            "Run `python model/train_models.py` to regenerate it."
        )
        st.stop()
    return joblib.load(artefact)


@st.cache_data(show_spinner="Reading uploaded CSV...")
def read_uploaded_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV, tolerating both ';' (raw UCI) and ',' delimiters.

    Cached on the file bytes so re-running the script on a widget change does
    not re-parse the upload.
    """
    from io import BytesIO

    # sep=None with the python engine sniffs the delimiter, which saves the user
    # from having to know that the original UCI file is semicolon-delimited.
    frame = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python")
    if frame.shape[1] == 1:
        raise ValueError(
            f"'{filename}' parsed into a single column - the delimiter could not "
            "be detected. Please upload a comma- or semicolon-separated CSV."
        )
    return frame


@st.cache_data(show_spinner="Scoring every model on this test set...")
def score_all_models(
    data_signature: str, features: pd.DataFrame, truth: pd.Series, threshold: float
) -> pd.DataFrame:
    """Build the six-model comparison table for the current test set.

    data_signature participates in the cache key so a new upload invalidates the
    table; features/truth are passed positionally for the actual computation.
    """
    rows = []
    for display_name in MODEL_REGISTRY:
        metrics, _ = score_at_threshold(
            load_model(display_name), features, truth, threshold
        )
        rows.append({"ML Model Name": display_name, **metrics})
    return pd.DataFrame(rows).set_index("ML Model Name")


def render_confusion_matrix(truth: pd.Series, predictions) -> None:
    matrix = confusion_frame(truth, predictions)
    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="BuPu",
        cbar=False,
        linewidths=1.2,
        linecolor="white",
        annot_kws={"size": 12},
        ax=axis,
    )
    axis.set_title("Confusion matrix", fontsize=11, pad=10)
    axis.tick_params(labelsize=9)
    plt.setp(axis.get_yticklabels(), rotation=0)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def render_roc_curve(truth: pd.Series, scores, display_name: str, auc: float) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(truth, scores)
    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color="#6A2C91",
        linewidth=2,
        label=f"{display_name} (AUC = {auc:.4f})",
    )
    axis.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1, label="Random (0.50)")
    axis.set_xlabel("False positive rate", fontsize=9)
    axis.set_ylabel("True positive rate", fontsize=9)
    axis.set_title("ROC curve", fontsize=11, pad=10)
    axis.tick_params(labelsize=8)
    axis.legend(loc="lower right", fontsize=8, frameon=False)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def ranked_bar_chart(
    values: pd.Series, value_label: str, category_label: str, highlight_best: bool = False
):
    """Horizontal bars that keep the order of `values` as given.

    st.bar_chart re-sorts its categorical axis alphabetically and ignores the
    order of the Series it is handed, which turns a ranking into an alphabetical
    list. Encoding the order explicitly through Altair's `sort` is the only
    reliable way to make the chart show what the numbers actually say.
    """
    frame = pd.DataFrame(
        {category_label: values.index.astype(str), value_label: values.to_numpy()}
    )
    category_order = frame[category_label].tolist()

    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3, height=18)
        .encode(
            x=alt.X(f"{value_label}:Q", title=value_label),
            y=alt.Y(f"{category_label}:N", sort=category_order, title=None),
            color=(
                alt.condition(
                    alt.datum[value_label] == float(values.max()),
                    alt.value("#6A2C91"),
                    alt.value("#C9B6DB"),
                )
                if highlight_best
                else alt.value("#6A2C91")
            ),
            tooltip=[category_label, alt.Tooltip(f"{value_label}:Q", format=".4f")],
        )
    )
    labels = bars.mark_text(align="left", dx=4, fontSize=11).encode(
        text=alt.Text(f"{value_label}:Q", format=".4f"), color=alt.value("#444444")
    )
    return (bars + labels).properties(height=max(160, 30 * len(frame)))


def render_feature_influence(pipeline, display_name: str) -> None:
    """Show the top drivers, using whichever interpretability hook the model has."""
    encoder = pipeline.named_steps["encode"]
    classifier = pipeline.named_steps["classify"]
    feature_names = [name.split("__", 1)[-1] for name in encoder.get_feature_names_out()]

    if hasattr(classifier, "feature_importances_"):
        caption = "Impurity-based feature importance (higher = more splits gained)."
        influence = pd.Series(classifier.feature_importances_, index=feature_names)
        ordered = influence.sort_values(ascending=False).head(12)
    elif hasattr(classifier, "coef_"):
        caption = (
            "Logistic-regression coefficients on standardised features. "
            "Positive pushes the prediction toward *subscribed*."
        )
        influence = pd.Series(classifier.coef_[0], index=feature_names)
        ordered = influence.reindex(influence.abs().sort_values(ascending=False).index).head(12)
    else:
        st.info(
            f"**{display_name}** exposes no coefficients or impurity importances, "
            "so per-feature influence cannot be read directly off the fitted model."
        )
        return

    st.caption(caption)
    st.altair_chart(
        ranked_bar_chart(ordered, "Influence", "Feature"), width="stretch"
    )


# ----------------------------------------------------------------------------
# Sidebar - data upload, model choice, decision threshold
# ----------------------------------------------------------------------------
st.sidebar.title("Controls")

st.sidebar.subheader("1. Test data")
uploaded = st.sidebar.file_uploader(
    "Upload a labelled test CSV",
    type=["csv"],
    help=(
        "Needs the original UCI Bank Marketing columns plus the 'y' label. "
        "Comma or semicolon delimited. Leave empty to use the bundled "
        "held-out split."
    ),
)

if uploaded is not None:
    try:
        test_frame = read_uploaded_csv(uploaded.getvalue(), uploaded.name)
    except ValueError as error:
        st.sidebar.error(str(error))
        st.stop()
    data_source = f"uploaded file `{uploaded.name}`"
    data_signature = f"{uploaded.name}:{uploaded.size}"
elif BUNDLED_TEST_CSV.exists():
    test_frame = pd.read_csv(BUNDLED_TEST_CSV)
    data_source = "bundled `test_data.csv` (the held-out 25% split)"
    data_signature = "bundled"
    st.sidebar.info("No upload yet - showing results on the bundled test split.")
else:
    st.error("No test data available. Upload a CSV or run the training script first.")
    st.stop()

try:
    features, truth = split_features_and_target(test_frame)
except ValueError as error:
    st.error(f"**Could not use this file.** {error}")
    st.stop()

st.sidebar.subheader("2. Model")
selected_model_name = st.sidebar.selectbox(
    "Choose a classifier",
    options=list(MODEL_REGISTRY),
    index=4,  # Random Forest - the best MCC on this dataset
    help="All six are already trained; switching re-scores the test set live.",
)

st.sidebar.subheader("3. Decision threshold")
threshold = st.sidebar.slider(
    "Classify as 'will subscribe' when probability >=",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help=(
        "0.50 is only a convention. Lower it to call more clients and catch more "
        "subscribers (higher recall, lower precision); raise it to waste fewer "
        "calls. AUC does not change - it already averages over all thresholds."
    ),
)

st.sidebar.divider()
st.sidebar.caption(
    "Dataset: UCI Bank Marketing (Moro, Cortez & Rita, 2014)\n\n"
    "Models trained offline on a stratified 75% split; this app only performs "
    "inference."
)

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.title("📞 Term Deposit Subscription Predictor")
st.markdown(
    "Which bank clients will subscribe to a **term deposit** after a marketing "
    "call? Six classifiers, one dataset, six metrics each — compared live on "
    "your test data."
)

positive_rate = truth.mean()
summary_columns = st.columns(4)
summary_columns[0].metric("Test rows", f"{len(features):,}")
summary_columns[1].metric("Features used", f"{features.shape[1]}")
summary_columns[2].metric("Subscribed ('yes')", f"{positive_rate:.1%}")
summary_columns[3].metric(
    "Majority-class baseline", f"{max(positive_rate, 1 - positive_rate):.1%}"
)
st.caption(
    f"Source: {data_source}. The baseline is the accuracy of blindly predicting "
    "the majority class — any model that cannot beat it has learned nothing, "
    "which is exactly why accuracy alone is a poor headline metric here."
)

single_tab, compare_tab, data_tab = st.tabs(
    ["Selected model", "All six models", "Test data"]
)

with single_tab:
    pipeline = load_model(selected_model_name)
    metrics, predictions = score_at_threshold(pipeline, features, truth, threshold)

    st.subheader(f"{selected_model_name} — evaluation metrics")
    if threshold != 0.50:
        st.caption(f"Computed at a decision threshold of **{threshold:.2f}**.")

    metric_columns = st.columns(6)
    for column, metric_name in zip(metric_columns, METRIC_ORDER):
        column.metric(
            metric_name, f"{metrics[metric_name]:.4f}", help=METRIC_HELP[metric_name]
        )

    st.divider()
    left_column, right_column = st.columns(2)
    with left_column:
        render_confusion_matrix(truth, predictions)
    with right_column:
        render_roc_curve(
            truth,
            positive_class_scores(pipeline, features),
            selected_model_name,
            metrics["AUC"],
        )

    st.subheader("Classification report")
    report = classification_report(
        truth,
        predictions,
        target_names=["no (did not subscribe)", "yes (subscribed)"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(
        pd.DataFrame(report).transpose().style.format("{:.4f}"),
        width="stretch",
    )

    with st.expander("What is driving this model's predictions?"):
        render_feature_influence(pipeline, selected_model_name)

with compare_tab:
    st.subheader("Comparison table — all six models on this test set")
    comparison = score_all_models(data_signature, features, truth, threshold)

    st.dataframe(
        comparison.style.format("{:.4f}").highlight_max(axis=0, color="#D6EFD8"),
        width="stretch",
    )
    st.caption(
        "Best value per metric is highlighted. Note that the winner differs by "
        "metric — the highest-accuracy model is **not** the best model here."
    )

    st.subheader("Ranking by metric")
    chosen_metric = st.radio(
        "Rank models by", METRIC_ORDER, index=5, horizontal=True,
        help="Pick a metric to re-rank the six models. The order changes with the "
             "metric, which is the whole point of the table above.",
    )
    st.caption(METRIC_HELP[chosen_metric])
    st.altair_chart(
        ranked_bar_chart(
            comparison[chosen_metric].sort_values(ascending=False),
            chosen_metric,
            "Model",
            highlight_best=True,
        ),
        width="stretch",
    )

    leader = comparison[chosen_metric].idxmax()
    st.success(
        f"**{leader}** leads on {chosen_metric} "
        f"({comparison.loc[leader, chosen_metric]:.4f}) at threshold {threshold:.2f}."
    )

    st.download_button(
        "Download this comparison table as CSV",
        data=comparison.round(4).to_csv().encode(),
        file_name=f"model_comparison_threshold_{threshold:.2f}.csv",
        mime="text/csv",
        help="Exports the six-model table exactly as computed on the current "
             "test set and threshold.",
    )

with data_tab:
    st.subheader("Test data preview")
    st.dataframe(test_frame.head(50), width="stretch")
    st.caption(f"Showing 50 of {len(test_frame):,} rows from the {data_source}.")

    st.subheader("Class balance")
    st.bar_chart(test_frame["y"].value_counts(), height=260)

    st.download_button(
        "Download this test set as CSV",
        data=test_frame.to_csv(index=False).encode(),
        file_name="test_data.csv",
        mime="text/csv",
    )
