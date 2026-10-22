import json

import joblib
import pandas as pd
import pytest

from app.agent.retrieval import PolicyRetriever, load_rules
from app.agent.tools import TOOL_SCHEMAS, ToolBox
from app.features import FEATURE_COLS


@pytest.fixture(scope="module")
def toolbox():
    # Same artefacts the app uses: committed model + demo sample
    df = pd.read_csv("data/sample/transactions.csv", parse_dates=["trans_date_trans_time"])
    model = joblib.load("models/model.joblib")
    df["score"] = model.predict_proba(df[FEATURE_COLS])[:, 1]
    return ToolBox(df, model, PolicyRetriever(load_rules()), threshold=0.02)


@pytest.fixture(scope="module")
def any_id(toolbox):
    return toolbox.df.index[0]


def test_agent_never_sees_the_label(toolbox, any_id):
    # Ground truth would let the agent "cheat"; it must not be in any tool output
    for name in ["get_transaction", "get_model_assessment"]:
        output = json.dumps(toolbox.call(name, {"trans_num": any_id}))
        assert "is_fraud" not in output


def test_outputs_are_valid_json(toolbox, any_id):
    # NaN is not valid JSON -> must be converted to None before reaching the LLM
    output = json.dumps(toolbox.call("get_model_assessment", {"trans_num": any_id}), allow_nan=False)
    assert "score" in output


@pytest.mark.parametrize("name, args", [
    ("get_transaction", {"trans_num": "does-not-exist"}),
    ("unknown_tool", {}),
    ("get_transaction", {"wrong_arg": 1}),
])
def test_bad_calls_return_error_not_crash(toolbox, name, args):
    assert "error" in toolbox.call(name, args)


def test_every_schema_has_a_tool(toolbox):
    for schema in TOOL_SCHEMAS:
        assert hasattr(toolbox, schema["function"]["name"])