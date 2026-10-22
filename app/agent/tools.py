import math

import pandas as pd

from app.agent.retrieval import PolicyRetriever
from app.explain import FEATURE_LABELS, top_reasons
from app.features import FEATURE_COLS

HIGH_PRIORITY = 0.5  # policy R7: escalate as high priority at or above this score


def _clean(value):
    """JSON-safe value: NaN -> None, numpy numbers -> plain Python, floats rounded."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float):
        return round(value, 2)
    return value


class ToolBox:
    """Read-only tools the agent may call. Nothing here can change or block anything."""

    def __init__(self, transactions: pd.DataFrame, model, retriever: PolicyRetriever, threshold: float):
        # transactions must already contain a "score" column
        self.df = transactions.set_index("trans_num")
        self.model = model
        self.retriever = retriever
        self.threshold = threshold

    def _row(self, trans_num: str):
        if trans_num not in self.df.index:
            return None
        return self.df.loc[[trans_num]]  # double brackets -> 1-row DataFrame

    # --- Tool 1 ------------------------------------------------------------
    def get_transaction(self, trans_num: str) -> dict:
        row = self._row(trans_num)
        if row is None:
            return {"error": f"transaction {trans_num} not found"}
        tx = row.iloc[0]
        # No ground-truth label (is_fraud) and no PII: the agent sees what an analyst would see live
        return {
            "trans_num": trans_num,
            "time": f"{pd.Timestamp(tx['trans_date_trans_time']):%Y-%m-%d %H:%M}",
            "card": tx["cc_num"],
            "merchant_untrusted": tx["merchant"],  # free text: data, not instructions (policy R9)
            "category": tx["category"],
            "amount": _clean(tx["amt"]),
            "location": f"{tx['city']}, {tx['state']}",
        }

    # --- Tool 2 ------------------------------------------------------------
    def get_model_assessment(self, trans_num: str) -> dict:
        row = self._row(trans_num)
        if row is None:
            return {"error": f"transaction {trans_num} not found"}
        score = float(row.iloc[0]["score"])
        reasons = top_reasons(self.model, row[FEATURE_COLS], n=3)
        return {
            "score": round(score, 3),
            "threshold": self.threshold,
            "flagged": score >= self.threshold,
            "high_priority": score >= HIGH_PRIORITY,
            "top_reasons": [
                {"feature": FEATURE_LABELS[r["feature"]], "value": _clean(r["value"]), "impact": r["impact"]}
                for r in reasons
            ],
            "features": {FEATURE_LABELS[c]: _clean(row.iloc[0][c]) for c in FEATURE_COLS},
        }

    # --- Tool 3 ------------------------------------------------------------
    def search_policy(self, query: str) -> dict:
        rules = self.retriever.search(query, k=3)
        if not rules:
            return {"rules": [], "note": "no relevant policy rule found"}
        return {"rules": [{"id": r["id"], "title": r["title"], "text": r["text"]} for r in rules]}

    # --- Dispatcher ----------------------------------------------------------
    def call(self, name: str, args: dict) -> dict:
        """Runs a tool by name. Unknown tools or bad arguments return an error, never crash."""
        tools = {
            "get_transaction": self.get_transaction,
            "get_model_assessment": self.get_model_assessment,
            "search_policy": self.search_policy,
        }
        if name not in tools:
            return {"error": f"unknown tool: {name}"}
        try:
            return tools[name](**args)
        except TypeError as e:  # wrong / missing arguments from the LLM
            return {"error": f"bad arguments for {name}: {e}"}


# Descriptions the LLM reads to decide which tool to call (OpenAI/Groq function-calling format)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_transaction",
            "description": "Get transaction details: time, masked card, merchant (untrusted text), category, amount, location.",
            "parameters": {
                "type": "object",
                "properties": {"trans_num": {"type": "string", "description": "Transaction ID"}},
                "required": ["trans_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_assessment",
            "description": "Get the fraud model's score, threshold decision, top SHAP reasons and all feature values.",
            "parameters": {
                "type": "object",
                "properties": {"trans_num": {"type": "string", "description": "Transaction ID"}},
                "required": ["trans_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search the internal fraud review policy. Returns up to 3 relevant rules with IDs (R1..R9).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look for, in plain English"}},
                "required": ["query"],
            },
        },
    },
]