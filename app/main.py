from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
from joblib import load
from pathlib import Path

app = FastAPI(title="Smoker Detection API")

# ── 1. Load trained models ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

model_path_catboost = BASE_DIR / "CatBoost_86.joblib"
model_path_xgboost = BASE_DIR / "xgb_model87.json"  # Make sure filename matches

model_catboost = load(model_path_catboost)
model_xgboost = load(model_path_xgboost)

# ── 2. Input schema ─────────────────────────────────────────────────────────────
class InputData(BaseModel):
    hdl:                     float
    ldl:                     float
    relaxation:              float
    fasting_blood_sugar:     float
    ast:                     float
    alt:                     float
    gtp:                     float
    serum_creatinine:        float
    hemoglobin:              float
    urine_protein:           float
    waist_cm:                float
    combined_hearing:        float
    combined_vision:         float
    systolic:                float
    cholesterol:             float
    triglyceride:            float
    dental_caries:           float
    age:                     int
    height_cm:               float
    weight_kg:               float

# ── 3. Mapping to model feature names ───────────────────────────────────────────
RENAME_TO_MODEL = {
    "hdl":                   "HDL",
    "ldl":                   "LDL",
    "relaxation":            "relaxation",
    "fasting_blood_sugar":   "fasting blood sugar",
    "ast":                   "AST",
    "alt":                   "ALT",
    "gtp":                   "Gtp",
    "serum_creatinine":      "serum creatinine",
    "hemoglobin":            "hemoglobin",
    "urine_protein":         "Urine protein",
    "waist_cm":              "waist(cm)",
    "combined_hearing":      "hearing_combined",
    "combined_vision":       "vision_combined",
    "systolic":              "systolic",
    "cholesterol":           "Cholesterol",
    "triglyceride":          "triglyceride",
    "dental_caries":         "dental caries",
    "age":                   "age",
    "height_cm":             "height(cm)",
    "weight_kg":             "weight(kg)"
}

# ── 4. Prediction endpoint ──────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: InputData, model_choice: str = Query("catboost", enum=["catboost", "xgboost"])):
    df = pd.DataFrame([data.dict()])
    df.rename(columns=RENAME_TO_MODEL, inplace=True)

    selected_model = model_catboost if model_choice == "catboost" else model_xgboost

    expected_features = selected_model.feature_names_
    missing = set(expected_features) - set(df.columns)
    if missing:
        return {"error": f"Missing features: {missing}"}

    df = df[expected_features]

    class_hat = selected_model.predict(df)
    proba_hat = selected_model.predict_proba(df)[:, 1]

    return {
        "model_used": model_choice,
        "prediction": int(class_hat[0]),
        "probability": float(proba_hat[0])
    }