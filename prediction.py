import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint
from typing import Literal

# Load trained models
rf_model = joblib.load("rf_tuned_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
lgbm_model = joblib.load("lgbm_model.pkl")

# Load encoders and feature order
label_encoders = joblib.load("label_encoders.pkl")
feature_order = joblib.load("feature_columns.pkl")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow your frontend
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define the input data model with validations
class PatientInfo(BaseModel):
    time_in_hospital: conint(ge=1, le=30)  # Days in hospital must be between 1 and 30
    n_lab_procedures: conint(ge=0, le=100)  # Number of lab procedures must be between 0 and 100
    n_procedures: conint(ge=0, le=10)        # Number of procedures must be between 0 and 10
    n_medications: conint(ge=0, le=50)       # Number of medications must be between 0 and 50
    age: Literal["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
                 "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
    glucose_test: Literal["Yes", "No"]
    A1Ctest: Literal["Yes", "No"]
    change: Literal["Yes", "No"]
    diabetes_med: Literal["Yes", "No"]
    additional_param1: Literal["Yes", "No"]  # New parameter
    additional_param2: Literal["Yes", "No"]  # New parameter

@app.post("/predict")
async def predict_readmission(patient_info: PatientInfo):
    try:
        # Convert user input into a DataFrame
        user_input_df = pd.DataFrame([patient_info.dict()])

        # Encode categorical features
        for col in label_encoders:
            if col in user_input_df:
                user_input_df[col] = user_input_df[col].apply(lambda x: x if x in label_encoders[col].classes_ else "Unknown")
                label_encoders[col].classes_ = np.append(label_encoders[col].classes_, "Unknown")
                user_input_df[col] = label_encoders[col].transform(user_input_df[col])

        # Ensure the feature order matches the training data
        user_input_df = user_input_df.reindex(columns=feature_order, fill_value=0)

        # Make predictions
        rf_prediction = rf_model.predict(user_input_df)[0]
        rf_proba = float(rf_model.predict_proba(user_input_df)[0][1])  # Convert to Python float

        xgb_prediction = xgb_model.predict(user_input_df)[0]
        xgb_proba = float(xgb_model.predict_proba(user_input_df)[0][1])

        lgbm_prediction = lgbm_model.predict(user_input_df)[0]
        lgbm_proba = float(lgbm_model.predict_proba(user_input_df)[0][1])

        # Format predictions
        def format_prediction(pred, proba):
            return {
                "status": "Not Likely to be Readmitted" if pred == 1 else "Likely to be Readmitted",
                "probability": float(proba)  # Convert NumPy float to Python float
            }

        rf_result = format_prediction(rf_prediction, rf_proba)
        xgb_result = format_prediction(xgb_prediction, xgb_proba)
        lgbm_result = format_prediction(lgbm_prediction, lgbm_proba)

        # Choose final prediction (majority vote)
        final_prediction = round((rf_prediction + xgb_prediction + lgbm_prediction) / 3)
        final_proba = float((rf_proba + xgb_proba + lgbm_proba) / 3)
        final_result = format_prediction(final_prediction, final_proba)

        return {
            "Random Forest": rf_result,
            "XGBoost": xgb_result,
            "LightGBM": lgbm_result,
            "Final Prediction": final_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# To run the app, use the command: uvicorn filename:app --host 127.0.0.1 --port 8000 --reload
