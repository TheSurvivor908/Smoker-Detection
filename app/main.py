from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from joblib import load
from pathlib import Path
import os
import uvicorn
import xgboost as xgb

app = FastAPI(title="Smoker Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"]
)

# ── 1. Load trained models ──────────────────────────────────────────────────────

# Set the base directory based on an environment variable
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent))

model_path_catboost = MODEL_DIR / "CatBoost_86.joblib"
model_path_xgboost = MODEL_DIR / "xgb_model87.json"

# Load models
try:
    model_catboost = load(model_path_catboost)
except Exception as e:
    logger.error(f"Failed to load CatBoost model: {e}")
    raise HTTPException(status_code=500, detail="Failed to load CatBoost model")

try:
    model_xgboost = xgb.Booster()
    model_xgboost.load_model(model_path_xgboost)
except Exception as e:
    logger.error(f"Failed to load XGBoost model: {e}")
    raise HTTPException(status_code=500, detail="Failed to load XGBoost model")

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
    "hearing_combined": "combined_hearing", 
    "vision_combined": "combined_vision",    
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
    # Create DataFrame from input data
    df = pd.DataFrame([data.dict()])

    # Apply the mapping to rename columns
    df.rename(columns=RENAME_TO_MODEL, inplace=True)

    if model_choice == "catboost":
        # Reorder columns to match model expectations
        expected_features = model_catboost.feature_names_
        missing = set(expected_features) - set(df.columns)
        if missing:
            return {"error": f"Missing features: {missing}"}
        
        df = df[expected_features]

        # Predict using CatBoost
        class_hat = model_catboost.predict(df)
        proba_hat = model_catboost.predict_proba(df)[:, 1]

    elif model_choice == "xgboost":
        # Ensure feature order for XGBoost if known (you may define a list manually)
        dmatrix = xgb.DMatrix(df)

        # Predict with Booster model
        proba_hat = model_xgboost.predict(dmatrix)
        class_hat = (proba_hat > 0.5).astype(int)  # Convert to class label manually

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

print(os.path.exists(MODEL_DIR / "CatBoost_86.joblib"))