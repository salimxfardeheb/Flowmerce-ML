import sys
import os

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

from config import (
    MODEL_RESOLUTION,
    MODEL_SHIPPING,
    OHE_ENCODER,
    SCALER,
    TRAIN_COLUMNS,
    TRAINING_PARAMS,
    RESOLUTION_LABELS,
    SHIPPING_LABELS,
)
from src.preprocessing import preprocess


# ═══════════════════════════════════════════════════════════════
#  CHARGEMENT DES ARTEFACTS
# ═══════════════════════════════════════════════════════════════
model_resolution = joblib.load(MODEL_RESOLUTION)
model_shipping   = joblib.load(MODEL_SHIPPING)
ohe              = joblib.load(OHE_ENCODER)
scaler           = joblib.load(SCALER)
train_columns    = joblib.load(TRAIN_COLUMNS)
training_params  = joblib.load(TRAINING_PARAMS)

seuil_risque = training_params["seuil_risque"]


# ═══════════════════════════════════════════════════════════════
#  SCHÉMA DE LA REQUÊTE
# ═══════════════════════════════════════════════════════════════
class ReturnRequest(BaseModel):
    Customer_Gender: str
    Customer_Age: int
    Customer_Wilaya: str
    Customer_Past_Returns: int = Field(ge=0)
    Shop_Name: str
    Product_Category: str
    Product_Price_DA: float = Field(gt=0)
    Order_Quantity: int = Field(ge=1)
    Total_Amount_DA: float = Field(gt=0)
    Payment_Method: str
    Shipping_Method: str
    Shipping_Cost_DA: float = Field(ge=0)
    Return_Reason: str
    Days_to_Return: int = Field(ge=0)
    Shop_Return_Window_Days: int = Field(gt=0)
    Within_Return_Policy: int = Field(ge=0, le=1)
    Fraud_Score: float = Field(ge=0, le=100)
    Customer_Satisfaction: int = Field(ge=1, le=5)
    Is_Suspicious: int = Field(ge=0, le=1)
    Refund_Amount_DA: float = Field(ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "Customer_Gender": "Female",
                "Customer_Age": 30,
                "Customer_Wilaya": "Alger",
                "Customer_Past_Returns": 3,
                "Shop_Name": "TechnoStore",
                "Product_Category": "Electronics",
                "Product_Price_DA": 15000.0,
                "Order_Quantity": 1,
                "Total_Amount_DA": 15500.0,
                "Payment_Method": "CCP",
                "Shipping_Method": "Standard",
                "Shipping_Cost_DA": 500.0,
                "Return_Reason": "Defective",
                "Days_to_Return": 5,
                "Shop_Return_Window_Days": 14,
                "Within_Return_Policy": 1,
                "Fraud_Score": 12.0,
                "Customer_Satisfaction": 4,
                "Is_Suspicious": 0,
                "Refund_Amount_DA": 15000.0,
            }
        }


# ═══════════════════════════════════════════════════════════════
#  APPLICATION FASTAPI
# ═══════════════════════════════════════════════════════════════
app = FastAPI(
    title="Flowmerce — API de Prédiction des Retours",
    description="Prédiction de la résolution d'un retour et du payeur des frais de retour.",
    version="2.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Flowmerce Returns Prediction API",
        "version": "2.0.0",
        "endpoints": {
            "/predict": "POST — Prédire la résolution et le payeur du retour",
            "/health": "GET  — Vérifier l'état de l'API",
        },
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": {
            "resolution": model_resolution is not None,
            "shipping": model_shipping is not None,
        },
        "artifacts_loaded": {
            "ohe_encoder": ohe is not None,
            "scaler": scaler is not None,
            "train_columns": train_columns is not None,
            "training_params": training_params is not None,
        },
        "seuil_risque": seuil_risque,
    }


@app.post("/predict")
def predict(request: ReturnRequest):
    try:
        row = pd.DataFrame([request.model_dump()])

        # Prétraitement complet avec le seuil appris
        X = preprocess(row, ohe, scaler, train_columns, seuil_risque=seuil_risque)

        # Prédictions
        pred_res  = model_resolution.predict(X)[0]
        pred_ship = model_shipping.predict(X)[0]

        # Probabilités
        proba_res  = model_resolution.predict_proba(X)[0]
        proba_ship = model_shipping.predict_proba(X)[0]

        return {
            "resolution": {
                "prediction": RESOLUTION_LABELS.get(pred_res, str(pred_res)),
                "probabilities": {
                    RESOLUTION_LABELS[i]: round(float(p), 4)
                    for i, p in enumerate(proba_res)
                },
            },
            "shipping_paid_by": {
                "prediction": SHIPPING_LABELS.get(pred_ship, str(pred_ship)),
                "probabilities": {
                    SHIPPING_LABELS[i]: round(float(p), 4)
                    for i, p in enumerate(proba_ship)
                },
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)