# app/main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
from joblib import load
from pathlib import Path

app = FastAPI(
    title="Health‑Risk Ensemble API",
    description="Serve predictions with CatBoost or XGBoost model via query switch",
    version="1.1.0"
)

# ── 1. Load trained models (RELATIVE paths, so it works on Railway/anywhere) ────
BASE_DIR = Path(__file__).resolve().parent
model_catboost = load(BASE_DIR / "project4_CatBoost_86.joblib")
model_xgboost  = load(BASE_DIR / "project4_XGBOOST_87.joblib")

# ── 2. Input schema (python‑safe names) ────────────────────────────────────────
class InputData(BaseModel):
    hdl:                   float
    ldl:                   float
    relaxation:            float
    fasting_blood_sugar:   float
    ast:                   float
    alt:                   float
    gtp:                   float
    serum_creatinine:      float
    hemoglobin:            float
    urine_protein:         float
    waist_cm:              float
    combined_hearing:      float
    combined_vision:       float
    systolic:              float
    cholesterol:           float
    triglyceride:          float
    dental_caries:         float
    age:                   int
    height_cm:             float
    weight_kg:             float

# map API field → original training column
RENAME_TO_MODEL = {
    "hdl":                 "HDL",
    "ldl":                 "LDL",
    "relaxation":          "relaxation",
    "fasting_blood_sugar": "fasting blood sugar",
    "ast":                 "AST",
    "alt":                 "ALT",
    "gtp":                 "Gtp",
    "serum_creatinine":    "serum creatinine",
    "hemoglobin":          "hemoglobin",
    "urine_protein":       "Urine protein",
    "waist_cm":            "waist(cm)",
    "combined_hearing":    "hearing_combined",
    "combined_vision":     "vision_combined",
    "systolic":            "systolic",
    "cholesterol":         "Cholesterol",
    "triglyceride":        "triglyceride",
    "dental_caries":       "dental caries",
    "age":                 "age",
    "height_cm":           "height(cm)",
    "weight_kg":           "weight(kg)"
}

# helper to choose model by query param
MODELS = {
    "catboost": model_catboost,
    "xgboost" : model_xgboost
}

# ── 3. Prediction endpoint ─────────────────────────────────────────────────────
@app.post("/predict")
def predict(
    data: InputData,
    model_choice: str = Query("catboost", enum=["catboost", "xgboost"])
):
    # 3‑a Convert request to DataFrame
    df = pd.DataFrame([data.dict()])

    # 3‑b Rename columns to original model names
    df.rename(columns=RENAME_TO_MODEL, inplace=True)

    # 3‑c Pick model
    selected_model = MODELS[model_choice]

    # 3‑d Ensure DataFrame has correct columns / order
    expected = selected_model.feature_names_
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        return {"error": f"Missing features: {missing_cols}"}

    df = df[expected]

    # 3‑e Predict
    class_hat = int(selected_model.predict(df)[0])
    proba_hat = float(selected_model.predict_proba(df)[:, 1][0])

    return {
        "model_used": model_choice,
        "prediction": class_hat,
        "probability": proba_hat
    }
