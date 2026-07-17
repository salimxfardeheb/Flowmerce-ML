"""
Configuration centralisée du projet Flowmerce.
Tous les chemins et constantes sont définis ici.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
#  CHEMINS
# ═══════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data
DATA_DIR         = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR     = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR    = os.path.join(DATA_DIR, "processed")
RAW_DATASET      = os.path.join(RAW_DATA_DIR, "ecommerce_returns_smart_dataset.csv")
RAW_DATASET_REAL = os.path.join(RAW_DATA_DIR, "ecommerce_returns_real_dataset.csv" )
SPLITS_FILE      = os.path.join(PROCESSED_DIR, "splits_encoded.pkl")

# Hugging Face Hub — si HF_DATASET_REPO est défini, le pipeline
# télécharge le dataset depuis le Hub au lieu du CSV local.
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")
HF_DATASET_FILE = os.environ.get("HF_DATASET_FILE", "ecommerce_returns_smart_dataset.csv")
HF_REPO_ID= os.environ.get("HF_REPO_ID")
HF_TOKEN        = os.environ.get("HF_TOKEN")
INTERNAL_KEY= os.environ.get("INTERNAL_API_KEY")

# Models
MODELS_DIR          = os.path.join(PROJECT_ROOT, "models")
MODEL_RESOLUTION    = os.path.join(MODELS_DIR, "model_resolution.joblib")
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

CSV_COLUMNS = [
    "Order_ID", "Customer_ID", "Customer_Age", "Customer_Gender",
    "Customer_Wilaya", "Customer_Past_Returns", "Shop_Name",
    "Product_Category", "Product_Name", "Product_Price_DA",
    "Order_Quantity", "Total_Amount_DA", "Payment_Method",
    "Shipping_Method", "Shipping_Cost_DA", "Order_Date", "Return_Date",
    "Days_to_Return", "Shop_Return_Window_Days", "Within_Return_Policy",
    "Return_Reason", "Resolution", "Return_Shipping_Paid_By",
    "Refund_Amount_DA", "Fraud_Score", "Is_Suspicious",
    "Customer_Satisfaction",
]


# ═══════════════════════════════════════════════════════════════
#  SEUILS DE PERFORMANCE
# ═══════════════════════════════════════════════════════════════
SEUIL_F1_RESOLUTION = 0.68
SEUIL_ACCURACY      = 0.70

PAYMENT_METHODS_ELECTRONIQUES = ["BaridiMob", "Carte Dahabia", "Edahabia", "CCP", "Virement"]


# ═══════════════════════════════════════════════════════════════
#  LABELS
# ═══════════════════════════════════════════════════════════════
RESOLUTION_MAP    = {"Exchange": 0, "Reject": 1, "Repair": 2}
RESOLUTION_LABELS = {0: "Exchange", 1: "Reject", 2: "Repair"}



# ═══════════════════════════════════════════════════════════════
#  PARAMÈTRES PAR DÉFAUT
# ═══════════════════════════════════════════════════════════════
SEUIL_NA           = 0.30
TEST_SIZE          = 0.2
RANDOM_STATE       = 42
PERCENTILE_RISQUE  = 75
N_ITER_SEARCH      = 5
