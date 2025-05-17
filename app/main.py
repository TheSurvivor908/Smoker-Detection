from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load

app = FastAPI(title="Smoker Status Detection API")

# ── 1. Load trained model ───────────────────────────────────────────────────────
model = load("C:/Users/ASUS/Dataset atau Handson/Project 4/Project 4 main model/app/project4_CatBoost_86.joblib")
        
# ── 2. Input schema with Python‑safe identifiers ────────────────────────────────
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

# ── 3. Mapping dict: API field  ➜  model's original column name ────────────────
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
    "combined_hearing":      "hearing_combined",      # ← adjust if different
    "combined_vision":       "vision_combined",       # ← adjust if different
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
def predict(data: InputData):
    # Convert pydantic → DataFrame
    df = pd.DataFrame([data.dict()])

    # Rename columns to match the model’s training schema
    df.rename(columns=RENAME_TO_MODEL, inplace=True)

    # Ensure column order equals model.feature_names_
    df = df[model.feature_names_]

    # Predict
    class_hat   = model.predict(df)
    proba_hat   = model.predict_proba(df)[:, 1]  # probability of positive class

    return {
        "prediction": int(class_hat[0]),
        "probability": float(proba_hat[0])
    }
