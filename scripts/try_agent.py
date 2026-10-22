import json

import joblib
import pandas as pd
from dotenv import load_dotenv

from app.agent.agent import investigate
from app.agent.retrieval import PolicyRetriever, load_rules
from app.agent.tools import ToolBox
from app.features import FEATURE_COLS

load_dotenv()

df = pd.read_csv("data/sample/transactions.csv", parse_dates=["trans_date_trans_time"])
model = joblib.load("models/model.joblib")
df["score"] = model.predict_proba(df[FEATURE_COLS])[:, 1]
toolbox = ToolBox(df, model, PolicyRetriever(load_rules()), threshold=0.02)

# The highest-scoring transaction: should end in "escalate"
trans_num = df.sort_values("score", ascending=False)["trans_num"].iloc[0]
result = investigate(trans_num, toolbox)

print("STATUS:", result["status"])
print("\nTOOLS CALLED:")
for a in result["audit"]:
    print(f"  step {a['step']}: {a['tool']}({a['args']})")
print("\nNOTE:\n", result["note"])