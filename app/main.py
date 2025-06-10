from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from joblib import load
from pathlib import Path
import os
import uvicorn
import xgboost as xgb
import logging

# ── Logger setup ────────────────────────────────────────────────────────────────
logger = logging.getLogger("smoker_api")
logging.basicConfig(level=logging.INFO)

# ── FastAPI app setup ───────────────────────────────────────────────────────────
app = FastAPI(title="Smoker Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://smoker-predict-api-production.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"]
)


# ── Load trained models ─────────────────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent))
model_path_catboost = MODEL_DIR / "CatBoost_86.joblib"
model_path_xgboost = MODEL_DIR / "xgb_model87.json"

try:
    model_catboost = load(model_path_catboost)
except Exception as e:
    logger.error(f"Failed to load CatBoost model: {e}")
    raise HTTPException(status_code=500, detail="Failed to load CatBoost model")

try:
    model_xgboost = xgb.Booster()
    model_xgboost.load_model(str(model_path_xgboost))
except Exception as e:
    logger.error(f"Failed to load XGBoost model: {e}")
    raise HTTPException(status_code=500, detail="Failed to load XGBoost model")

# ── Input schema ────────────────────────────────────────────────────────────────
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

# ── Feature mapping ─────────────────────────────────────────────────────────────
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
    "combined_hearing": "combined_hearing", 
    "combined_vision": "combined_vision",    
    "systolic": "systolic",
    "cholesterol": "Cholesterol",
    "triglyceride": "triglyceride",
    "dental_caries": "dental caries",
    "age": "age",
    "height_cm": "height(cm)",
    "weight_kg": "weight(kg)"
}

# ── Prediction endpoint ─────────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: InputData, model_choice: str = Query("catboost", enum=["catboost", "xgboost"])):
    df = pd.DataFrame([data.dict()])

    # Rename columns to match model expectations
    df.rename(columns=RENAME_TO_MODEL, inplace=True)

    if model_choice == "catboost":
        expected_features = model_catboost.feature_names_
        missing = set(expected_features) - set(df.columns)
        if missing:
            return {"error": f"Missing features: {missing}"}
        
        df = df[expected_features]

        class_hat = model_catboost.predict(df)
        proba_hat = model_catboost.predict_proba(df)[:, 1]

    elif model_choice == "xgboost":
        try:
            dmatrix = xgb.DMatrix(df.values, feature_names=df.columns.tolist())
            proba_hat = 0 #model_xgboost.predict(dmatrix)
            class_hat = (proba_hat > 0.5).astype(int)
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return {
        "model_used": model_choice,
        "prediction": int(class_hat[0]),
        "probability": float(proba_hat[0])
    }

# ── Basic routes ────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Welcome to the Smoker Detection API. Use the /predict endpoint to get started."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ── Run server ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)