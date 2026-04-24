"""
Configuration centralisée du projet Flowmerce.
Tous les chemins et constantes sont définis ici.
"""

import os

# ═══════════════════════════════════════════════════════════════
#  CHEMINS
# ═══════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data
DATA_DIR         = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR     = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR    = os.path.join(DATA_DIR, "processed")
RAW_DATASET      = os.path.join(RAW_DATA_DIR, "ecommerce_returns_smart_dataset.csv")
SPLITS_FILE      = os.path.join(PROCESSED_DIR, "splits_encoded.pkl")

# Models
MODELS_DIR          = os.path.join(PROJECT_ROOT, "models")
MODEL_RESOLUTION    = os.path.join(MODELS_DIR, "model_resolution.joblib")
MODEL_SHIPPING      = os.path.join(MODELS_DIR, "model_shipping.joblib")
OHE_ENCODER         = os.path.join(MODELS_DIR, "ohe_encoder.joblib")
SCALER              = os.path.join(MODELS_DIR, "scaler.joblib")
TRAIN_COLUMNS       = os.path.join(MODELS_DIR, "train_columns.joblib")
TRAINING_PARAMS     = os.path.join(MODELS_DIR, "training_params.joblib")


# ═══════════════════════════════════════════════════════════════
#  COLONNES
# ═══════════════════════════════════════════════════════════════
COLONNES_A_SUPPRIMER = [
    "Order_ID",
    "Customer_ID",
    "Product_Name",
    "Order_Date",
    "Return_Date",
    "Refund_Amount_DA",
    "Customer_Satisfaction",
    'Return_Shipping_Paid_By' 
]

COLONNES_CATEGORIEL = [
    "Customer_Gender",
    "Customer_Wilaya",
    "Shop_Name",
    "Product_Category",
    "Payment_Method",
    "Shipping_Method",
    "Return_Reason",
    "reason_x_policy",
]


# ═══════════════════════════════════════════════════════════════
#  SEUILS DE PERFORMANCE
# ═══════════════════════════════════════════════════════════════
SEUIL_F1_RESOLUTION = 0.62
SEUIL_ACCURACY      = 0.65


# ═══════════════════════════════════════════════════════════════
#  LABELS
# ═══════════════════════════════════════════════════════════════
RESOLUTION_MAP    = {"Exchange": 0, "Refund": 1, "Reject": 2, "Repair": 3}
RESOLUTION_LABELS = {v: k for k, v in RESOLUTION_MAP.items()}



# ═══════════════════════════════════════════════════════════════
#  PARAMÈTRES PAR DÉFAUT
# ═══════════════════════════════════════════════════════════════
SEUIL_NA           = 0.30
TEST_SIZE          = 0.2
RANDOM_STATE       = 42
PERCENTILE_RISQUE  = 75
N_ITER_SEARCH      = 30