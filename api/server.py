import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field
import joblib
import pandas as pd

from config import (
    MODEL_RESOLUTION,
    OHE_ENCODER,
    SCALER,
    TRAIN_COLUMNS,
    TRAINING_PARAMS,
    RESOLUTION_LABELS,
)
from src.preprocessing import preprocess

from dotenv import load_dotenv



# ═══════════════════════════════════════════════════════════════
#  CHARGEMENT DES ARTEFACTS
# ═══════════════════════════════════════════════════════════════
model_resolution = joblib.load(MODEL_RESOLUTION)
ohe              = joblib.load(OHE_ENCODER)
scaler           = joblib.load(SCALER)
train_columns    = joblib.load(TRAIN_COLUMNS)
training_params  = joblib.load(TRAINING_PARAMS)

seuil_risque = training_params["seuil_risque"]


# ═══════════════════════════════════════════════════════════════
#  AUTHENTIFICATION — X-Internal-Key
# ═══════════════════════════════════════════════════════════════
load_dotenv()
INTERNAL_KEY   = os.environ.get("INTERNAL_API_KEY", "X-internal-key")
api_key_header = APIKeyHeader(name="X-Internal-Key", auto_error=False)

def verify_internal_key(api_key: str = Security(api_key_header)):
    if api_key != INTERNAL_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé interne invalide ou manquante."
        )
    return api_key


# ═══════════════════════════════════════════════════════════════
#  SCHÉMA DE LA REQUÊTE
#  - Customer_Satisfaction retiré (data leakage)
#  - Is_Suspicious calculé automatiquement depuis Fraud_Score
# ═══════════════════════════════════════════════════════════════
class ReturnRequest(BaseModel):
    Customer_Gender:         str
    Customer_Age:            int
    Customer_Wilaya:         str
    Customer_Past_Returns:   int   = Field(ge=0)
    Shop_Name:               str
    Product_Category:        str
    Product_Price_DA:        float = Field(gt=0)
    Order_Quantity:          int   = Field(ge=1)
    Total_Amount_DA:         float = Field(gt=0)
    Payment_Method:          str
    Shipping_Method:         str
    Shipping_Cost_DA:        float = Field(ge=0)
    Return_Reason:           str
    Days_to_Return:          int   = Field(ge=0)
    Shop_Return_Window_Days: int   = Field(gt=0)
    Within_Return_Policy:    int   = Field(ge=0, le=1)
    Fraud_Score:             float = Field(ge=0, le=100)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "Customer_Gender":         "Female",
            "Customer_Age":            34,
            "Customer_Wilaya":         "Alger",
            "Customer_Past_Returns":   1,
            "Shop_Name":               "Shop_001",
            "Product_Category":        "Vetements",
            "Product_Price_DA":        3500.0,
            "Order_Quantity":          1,
            "Total_Amount_DA":         3500.0,
            "Payment_Method":          "Especes livraison",
            "Shipping_Method":         "Yalidine",
            "Shipping_Cost_DA":        400.0,
            "Return_Reason":           "Mauvaise taille",
            "Days_to_Return":          4,
            "Shop_Return_Window_Days": 14,
            "Within_Return_Policy":    1,
            "Fraud_Score":             5.0,
        }
    })


# ═══════════════════════════════════════════════════════════════
#  APPLICATION FASTAPI
# ═══════════════════════════════════════════════════════════════
app = FastAPI(
    title="Flowmerce — API de Prediction des Retours",
    description="Prediction de la resolution d'un retour produit.",
    version="3.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Flowmerce Returns Prediction API",
        "version": "3.0.0",
        "endpoints": {
            "/predict": "POST — Predire la resolution du retour",
            "/health":  "GET  — Verifier l'etat de l'API",
        },
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": {
            "resolution": model_resolution is not None,
        },
        "artifacts_loaded": {
            "ohe_encoder":     ohe             is not None,
            "scaler":          scaler          is not None,
            "train_columns":   train_columns   is not None,
            "training_params": training_params is not None,
        },
        "seuil_risque": seuil_risque,
    }


@app.post("/predict")
def predict(
    request: ReturnRequest,
    _: str = Security(verify_internal_key),
):
    try:
        row = pd.DataFrame([request.model_dump()])

        # Is_Suspicious calcule automatiquement depuis Fraud_Score
        row["Is_Suspicious"] = (row["Fraud_Score"] >= 60).astype(int)

        # Pretraitement
        X = preprocess(row, ohe, scaler, train_columns, seuil_risque=seuil_risque)

        # Prediction
        pred_res  = model_resolution.predict(X)[0]
        proba_res = model_resolution.predict_proba(X)[0]

        resolution_label = RESOLUTION_LABELS.get(pred_res, str(pred_res))
        confidence       = round(float(max(proba_res)), 4)

        return {
            "resolution": {
                "prediction":    resolution_label,
                "confidence":    confidence,
                "probabilities": {
                    RESOLUTION_LABELS[i]: round(float(p), 4)
                    for i, p in enumerate(proba_res)
                },
            },
            "risk_flag": {
                "is_suspicious":   bool(row["Is_Suspicious"].iloc[0]),
                "fraud_score":     float(row["Fraud_Score"].iloc[0]),
                "seuil_risque":    seuil_risque,
                "above_threshold": bool(row["Fraud_Score"].iloc[0] >= seuil_risque),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)