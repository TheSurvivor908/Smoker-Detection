from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from joblib import load
from pathlib import Path
import os
import uvicorn

app = FastAPI(title="Smoker Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 1. Load trained models ──────────────────────────────────────────────────────

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app"))

model_path_catboost = MODEL_DIR / "CatBoost_86.joblib"
model_path_xgboost = MODEL_DIR / "xgb_model87.json"

# Here, actually load your models
model_catboost = load(model_path_catboost)
# For xgboost, you might load differently (e.g. xgb.Booster) - make sure to load correctly
# model_xgboost = ...

# ── 2. Input schema ─────────────────────────────────────────────────────────────
class InputData(BaseModel):
    hdl: float
    ldl: float
    relaxation: float
    fasting_blood_sugar: float
    ast: float
    alt: float
    gtp: float
    serum_creatinine: float
    hemoglobin: float
    urine_protein: float
    waist_cm: float
    combined_hearing: float
    combined_vision: float
    systolic: float
    cholesterol: float
    triglyceride: float
    dental_caries: float
    age: int
    height_cm: float
    weight_kg: float

# ── 3. Mapping to model feature names ───────────────────────────────────────────
RENAME_TO_MODEL = {
    "hdl": "HDL",
    "ldl": "LDL",
    "relaxation": "relaxation",
    "fasting_blood_sugar": "fasting blood sugar",
    "ast": "AST",
    "alt": "ALT",
    "gtp": "Gtp",
    "serum_creatinine": "serum creatinine",
    "hemoglobin": "hemoglobin",
    "urine_protein": "Urine protein",
    "waist_cm": "waist(cm)",
    "combined_hearing": "hearing_combined",
    "combined_vision": "vision_combined",
    "systolic": "systolic",
    "cholesterol": "Cholesterol",
    "triglyceride": "triglyceride",
    "dental_caries": "dental caries",
    "age": "age",
    "height_cm": "height(cm)",
    "weight_kg": "weight(kg)"
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Smoker Detection API. Use the /predict endpoint to get started."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
