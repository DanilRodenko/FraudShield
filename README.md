# FraudShield 🛡️

A machine learning system for detecting fraudulent credit card transactions, built as a portfolio project demonstrating end-to-end Data Science and ML Engineering skills.

---

## Project Structure
```
FraudShield/
│
├── data/
│   ├── raw/              # Original dataset (not tracked)
│   ├── processed/        # Cleaned and engineered features
│   └── splits/           # Train/test splits
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature Engineering & Train/Test Split
│   ├── 03_modeling.ipynb             # Model Training & Comparison
│   └── 04_explainability.ipynb       # SHAP Explainability
│
├── app/
│   ├── streamlit_app.py  # Streamlit UI
│   └── fastapi_app.py    # FastAPI REST endpoint
│
├── models/               # Saved model artifacts
├── requirements.txt
└── .gitignore
```

---

## Dataset
**Kaggle Credit Card Fraud Detection** — 284,807 transactions, 492 fraud cases (0.17%).

Features V1–V28 are PCA-transformed for confidentiality. Additional features: `Time`, `Amount`.

Download: [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) → place `creditcard.csv` in `data/raw/`.

---

## Key Findings (EDA)
- Severe class imbalance: only **0.17% fraud rate** — accuracy is not a valid metric
- Fraudulent transactions have lower median amount ($9 vs $22 for legitimate)
- Fraud peaks at **night (0–5h)** and **morning (10–11h)**
- Most informative features: **V17, V14, V12** (negative correlation) and **V11, V4, V2** (positive)
- Engineered features: `Hour`, `is_night`, `Time_scaled`, `Amount_scaled`

---

## Feature Engineering
- `StandardScaler` applied to `Amount` and `Time` (V1–V28 already scaled via PCA)
- Engineered `Hour` from `Time` and binary `is_night` flag
- Train/test split: **80/20** with `stratify=y` to preserve class balance ratio

---

## Models & Results

| Metric | Logistic Regression | Random Forest | XGBoost | XGBoost Tuned |
|--------|---------------------|---------------|---------|---------------|
| F1 | 0.108 | **0.885** | 0.860 | 0.850 |
| Precision | 0.058 | **0.953** | 0.874 | 0.863 |
| Recall | **0.918** | 0.827 | 0.847 | 0.837 |
| ROC AUC | 0.974 | 0.913 | 0.923 | **0.982** |

**Selected model: Random Forest** — best F1 and Precision with default parameters.

Class imbalance handled via:
- `class_weight='balanced'` for Logistic Regression
- `scale_pos_weight` for XGBoost
- Random Forest handles imbalance natively

XGBoost hyperparameters tuned via `GridSearchCV` (cv=3, scoring='f1'):
- Best params: `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`

---

## Explainability (SHAP)
SHAP (SHapley Additive exPlanations) used to interpret Random Forest predictions.

- **Global importance**: `Time_scaled`, `Hour`, `V14`, `V4`, `V10` are top features
- **V14**: low values strongly push predictions toward fraud
- **Force plot**: individual fraud transaction explained — V17=-6.5, V14=-6.2, V12=-4.7 all contribute strongly to fraud prediction
- **Key insight**: SHAP revealed non-linear temporal patterns in `Time` and `Hour` not captured by simple correlation

---

## API Usage

### FastAPI
```bash
uvicorn app.fastapi_app:app --reload --port 8080
```
Docs available at: `http://localhost:8080/docs`

Response format:
```json
{
  "prediction": 1,
  "probability": 0.53
}
```

### Streamlit
```bash
streamlit run app/streamlit_app.py
```

---

## Setup
```bash
git clone https://github.com/DanilRodenko/FraudShield.git
cd FraudShield
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Stack
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **ML**: scikit-learn, XGBoost
- **Explainability**: SHAP
- **API**: FastAPI, uvicorn
- **App**: Streamlit

---

## Author
Built by [Danil Rodenko](https://linkedin.com/in/danilrodenko) — Data Science portfolio project.
