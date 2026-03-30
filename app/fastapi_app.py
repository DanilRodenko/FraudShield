import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Hour: float
    is_night: int
    Time_scaled: float
    Amount_scaled: float

@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    return {"prediction": int(prediction), "probability": round(float(probability), 4)}