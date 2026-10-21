import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="About · FraudShield", page_icon="🛡️", layout="wide")


@st.cache_data
def load_metrics() -> dict:
    return json.loads(Path("models/metrics.json").read_text())


m = load_metrics()
thr = m["threshold"]
t = thr["test"]  # confusion-matrix numbers on test at the chosen threshold

st.title("About FraudShield")
st.markdown(
    "A card-fraud detection model with explanations for analysts. "
    "It scores each transaction, flags the risky ones and shows **why** — "
    "so a reviewer can decide quickly instead of trusting a black box."
)

# --- Results ------------------------------------------------------------------
st.header("Results on the test set")
st.caption(
    "Test = the last 6 months of data (Jun–Dec 2020), never seen during training, "
    "model selection or threshold choice. Used once."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("PR-AUC", f"{m['test_metrics']['pr_auc']:.3f}", help="Main metric under heavy imbalance. Random model ≈ 0.004.")
c2.metric("Recall", f"{t['recall']:.0%}", help="Share of fraud the model catches at the chosen threshold.")
c3.metric("Precision", f"{t['precision']:.1%}", help="Share of alerts that are real fraud.")
c4.metric("Threshold", f"{thr['value']}", help="Chosen on validation by minimum cost of errors.")

st.markdown(
    f"Out of **{t['tp'] + t['fn']:,}** frauds in the test period the model caught **{t['tp']:,}** "
    f"and missed **{t['fn']:,}**, raising **{t['fp']:,}** false alerts "
    f"({t['fp'] / (t['fp'] + t['tn']):.1%} of legitimate transactions)."
)

# --- How it works ---------------------------------------------------------------
st.header("How it works")
st.markdown(
    """
1. **Features** from the card's own history — *only past transactions*, so the model never sees the future:
   amount vs the card's usual spend, transactions in the last hour / 24h, night-time, online purchase, age, distance.
2. **Model** — XGBoost, chosen over logistic regression, a decision tree and LightGBM on a time-based validation set
   (PR-AUC 0.80 vs 0.36 for the baseline).
3. **Threshold** — not the default 0.5. Chosen by cost: a missed fraud ≈ **$%d** (average fraud amount),
   a false alert ≈ **$%d** (assumption). On validation this cut the cost of errors by ~69%% vs 0.5.
4. **Explanations** — SHAP values for every transaction: which features pushed the score up.
"""
    % (thr["cost_fn"], thr["cost_fp"])
)

# --- Explainability ---------------------------------------------------------------
st.header("What drives the model")
st.image(
    "reports/shap_summary.png",
    caption="SHAP summary on 5,000 test transactions. Each dot is a transaction; "
    "right = pushes towards fraud, colour = feature value (red = high).",
)
st.markdown(
    "Amount dominates, followed by night-time. A burst of transactions in the last hour raises risk, "
    "while many transactions over 24h lowers it — an active customer looks different from a sudden burst."
)

# --- Limitations ------------------------------------------------------------------
st.header("Limitations")
st.markdown(
    """
- **Synthetic data** (Sparkov simulator). Patterns are cleaner than in real payments — e.g. the night-time
  effect (~25x day vs night) is likely a generator artefact. A real model should not rely on it this heavily.
- **Cost of a false alert is an assumption** ($10). A different value moves the optimal threshold.
- **Low precision** (~1 in 6 alerts is real) means a lot of manual review. In production a two-tier policy
  would help: auto-block above a high threshold, send the middle band to analysts.
- **Drift:** PR-AUC drops from 0.80 (validation) to 0.73 (later test period). The model needs monitoring
  and periodic retraining.
- **Correlated features** (amount, card's usual amount, ratio) share SHAP credit — individual attributions
  should not be over-interpreted.
"""
)

# --- Data & privacy ------------------------------------------------------------------
st.header("Data & privacy")
st.markdown(
    """
- Cardholder name, address, date of birth, gender and job are **not used** by the model and **not shipped** with the app.
- Card numbers are masked to the last 4 digits.
- The demo sample oversamples fraud (~13%) so the queue has something to show; the real rate is 0.39%.
"""
)