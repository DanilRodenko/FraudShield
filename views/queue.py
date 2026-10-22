import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from app.explain import FEATURE_LABELS, top_reasons
from app.features import FEATURE_COLS

from app.agent.agent import investigate
from app.agent.guardrails import validate
from app.agent.retrieval import PolicyRetriever, load_rules
from app.agent.tools import ToolBox

# --- Loading (cached: runs once per server, not on every click) ---------------
@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")


@st.cache_data
def load_metrics() -> dict:
    return json.loads(Path("models/metrics.json").read_text())


@st.cache_data
def load_transactions() -> pd.DataFrame:
    df = pd.read_csv("data/sample/transactions.csv", parse_dates=["trans_date_trans_time"])
    # Score every transaction once; features were precomputed with the full card history
    df["score"] = load_model().predict_proba(df[FEATURE_COLS])[:, 1]
    return df


model = load_model()
metrics = load_metrics()
df = load_transactions()
threshold = metrics["threshold"]["value"]
df["flagged"] = df["score"] >= threshold

# --- Header ---------------------------------------------------------------------
st.title("🛡️ FraudShield — fraud review queue")
st.caption(
    f"XGBoost model · alert threshold {threshold} (chosen on validation by cost of errors). "
    "Demo sample from the test period: fraud is **oversampled** (~13% here vs 0.39% in reality)."
)

col1, col2, col3 = st.columns(3)
col1.metric("Transactions in sample", f"{len(df):,}")
col2.metric("Flagged for review", f"{df['flagged'].sum():,}")
col3.metric("Test recall at this threshold", f"{metrics['threshold']['test']['recall']:.0%}")

# --- Queue ----------------------------------------------------------------------
st.subheader("Review queue")
only_flagged = st.toggle("Show flagged only", value=True)
queue = df[df["flagged"]] if only_flagged else df
queue = queue.sort_values("score", ascending=False)

st.dataframe(
    queue[["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "score"]],
    hide_index=True,
    width="stretch",
    column_config={
        "trans_date_trans_time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
        "cc_num": "Card",
        "merchant": "Merchant",
        "category": "Category",
        "amt": st.column_config.NumberColumn("Amount", format="$%.2f"),
        "score": st.column_config.ProgressColumn("Fraud score", min_value=0.0, max_value=1.0, format="%.2f"),
    },
)

# --- Transaction detail ----------------------------------------------------------
st.subheader("Investigate a transaction")
if queue.empty:
    st.info("Nothing in the queue.")
    st.stop()

selected = st.selectbox(
    "Transaction",
    queue["trans_num"],
    format_func=lambda t: (
        f"{queue.loc[queue.trans_num == t, 'merchant'].iloc[0]} · "
        f"${queue.loc[queue.trans_num == t, 'amt'].iloc[0]:,.2f} · "
        f"score {queue.loc[queue.trans_num == t, 'score'].iloc[0]:.2f}"
    ),
)
row = queue[queue["trans_num"] == selected]
tx = row.iloc[0]

left, right = st.columns(2)

with left:
    verdict = "🚩 Flag for review" if tx["flagged"] else "✅ Pass"
    st.metric("Fraud score", f"{tx['score']:.2f}", verdict, delta_color="off")
    st.write(
        f"**Card:** {tx['cc_num']}  \n"
        f"**Merchant:** {tx['merchant']} ({tx['category']})  \n"
        f"**Amount:** ${tx['amt']:,.2f}  \n"
        f"**Time:** {tx['trans_date_trans_time']:%Y-%m-%d %H:%M}  \n"
        f"**Location:** {tx['city']}, {tx['state']}"
    )
    # Ground truth is only available because this is historical demo data
    st.caption(f"Actual label (demo only): {'fraud' if tx['is_fraud'] else 'legitimate'}")

with right:
    st.markdown("**Top reasons (SHAP)** — what pushed the score towards fraud")
    reasons = top_reasons(model, row[FEATURE_COLS], n=3)
    for r in reasons:
        label = FEATURE_LABELS.get(r["feature"], r["feature"])
        value = "no history" if pd.isna(r["value"]) else f"{r['value']:,.2f}"
        st.write(f"- **{label}** = {value} → impact **{r['impact']:+.2f}**")
    st.caption("Impact is in log-odds: positive values push towards fraud.")


# --- AI investigator ---------------------------------------------------------------
@st.cache_resource
def load_retriever() -> PolicyRetriever:
    return PolicyRetriever(load_rules())


@st.cache_data
def load_policy_text() -> str:
    return Path("docs/fraud_policy.md").read_text()


st.subheader("AI investigator")
st.caption("Drafts a note from the model output and the review policy. Recommendation only — the analyst decides.")

# Streamlit reruns the script on every click, so results are kept per transaction in session_state
investigations = st.session_state.setdefault("investigations", {})

if st.button("🔎 Investigate with AI"):
    toolbox = ToolBox(df, model, load_retriever(), threshold)
    result, problem = None, None
    with st.spinner("Agent is checking the transaction, the model and the policy..."):
        try:
            result = investigate(selected, toolbox)
        except ValueError as e:  # missing API key
            problem = f"AI investigator is not configured: {e}"
        except Exception as e:  # network / rate limit: report it, never crash the page
            problem = f"Agent call failed ({type(e).__name__}). Try again in a minute."
    # Messages are shown after the spinner: anything drawn inside it disappears when it closes
    if problem:
        st.warning(problem)
    else:
        check = validate(result["note"], result["audit"], load_policy_text())
        investigations[selected] = {"result": result, "check": check}

if selected in investigations:
    result = investigations[selected]["result"]
    check = investigations[selected]["check"]

    if check["ok"]:
        st.success(f"Recommendation: **{check['recommendation']}** · all guardrail checks passed")
        # Escape "$": Streamlit markdown treats text between two $ signs as a LaTeX formula
        st.markdown(check["note"].replace("$", r"\$").replace("\n", "  \n"))
    else:
        # A note that failed validation is not shown as a finding
        st.error("The agent's note failed validation and is not shown as a finding.")
        for err in check["errors"]:
            st.write(f"- {err}")
        if check["note"]:
            with st.expander("Unvalidated draft (for debugging)"):
                st.text(check["note"])

    with st.expander(f"Audit log · {len(result['audit'])} tool calls · status: {result['status']}"):
        for a in result["audit"]:
            st.markdown(f"**Step {a['step']} · {a['tool']}** `{a['args']}`")
            st.json(a["result"], expanded=False)