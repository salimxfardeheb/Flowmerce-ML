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

# Secret partagé web app ↔ API ML (en-tête X-Internal-Key).
# `None` signifie « non configuré » : l'API refuse alors de démarrer et rejette
# toute requête (fail-closed). Une chaîne vide ou blanche vaut « non configuré »
# — sinon `X-Internal-Key: ` (vide) authentifierait.
_cle_interne_brute = os.environ.get("INTERNAL_API_KEY")
INTERNAL_KEY = _cle_interne_brute.strip() if _cle_interne_brute and _cle_interne_brute.strip() else None

# Longueur minimale attendue du secret. En dessous, l'API démarre mais journalise
# un avertissement : un secret court est devinable par force brute.
LONGUEUR_MIN_CLE_INTERNE = 16

# Contrat de features — schéma partagé web app ↔ API ML.
# Généré depuis les artefacts entraînés (cf. src/feature_contract.py), versionné
# dans le dépôt, et copié à l'identique côté web app (lib/ml/feature-contract.json).
CONTRACTS_DIR    = os.path.join(PROJECT_ROOT, "contracts")
FEATURE_CONTRACT = os.path.join(CONTRACTS_DIR, "feature_contract.json")

# En-tête par lequel un appelant déclare la version de contrat sur laquelle il a
# été construit. Une version différente de celle des artefacts chargés = refus
# explicite, au lieu d'une prédiction silencieusement fausse.
ENTETE_VERSION_CONTRAT = "X-Feature-Contract-Version"

# Politique par feature catégorielle face à une valeur hors vocabulaire.
#   "alert"    → anomalie : le vocabulaire émis en production a divergé de celui
#                appris. Journalisé en warning et remonté dans la réponse.
#   "expected" → divergence structurelle connue et documentée : le vocabulaire
#                appris ne peut pas couvrir la production (Shop_Name est une
#                liste de 80 boutiques synthétiques ; aucun vendeur réel n'y
#                figure). Remonté dans la réponse, sans alerte.
POLITIQUE_CATEGORIES_INCONNUES = {
    # Vocabulaire simulé (Shop_001…Shop_080) : aucun vendeur réel n'y figure.
    # Feature retirée du modèle au prochain réentraînement.
    "Shop_Name": "expected",
    # Feature retirée du modèle au prochain réentraînement : la web app
    # transmet désormais le transporteur tel quel, sans le contraindre au
    # vocabulaire appris. Alerter là-dessus n'apprendrait plus rien.
    "Shipping_Method": "expected",
}
POLITIQUE_CATEGORIE_DEFAUT = "alert"

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
# Colonnes retirées avant l'entraînement.
# `Refund_Amount_DA` et `Return_Shipping_Paid_By` ne sont plus collectés ni
# écrits par /save_claim (absents de CSV_COLUMNS), mais restent listés ici :
# les datasets historiques (dont le dataset synthétique sur HF) les contiennent
# encore, et le drop est conditionnel — il garantit qu'ils ne repassent jamais
# en feature. Ne pas retirer de cette liste tant qu'un dataset les porte.
#
# `Label_Source` est retirée après le filtrage des labels (cf. pipeline.py) :
# elle qualifie la cible, elle n'est jamais une feature.
#
# `Shop_Name` et `Shipping_Method` sont retirées du modèle (décision du
# 2026-08-11). Elles restent collectées par /save_claim — le dataset doit
# continuer de les porter, elles décrivent la réalité d'une réclamation — mais
# elles ne sont plus encodées comme features :
#
#   Shop_Name        cardinalité non bornée, qui croît avec le nombre de
#                    vendeurs. Chaque nouvelle boutique était hors vocabulaire à
#                    vie : le one-hot ne peut pas apprendre une liste ouverte.
#                    Le signal utile d'une boutique (taux de retour historique,
#                    ancienneté, volume) se porte par des features numériques,
#                    qui ne périment pas.
#   Shipping_Method  vocabulaire vivant (les transporteurs algériens
#                    apparaissent et disparaissent), pour un apport marginal :
#                    le rapport d'entraînement plaçait `Shipping_Method_Yalidine`
#                    au 19e rang des importances.
#
# La suppression prend effet au prochain entraînement. L'inférence suit
# automatiquement : `src/preprocessing.encoder_features` lit les colonnes à
# encoder sur `ohe.feature_names_in_`, donc sur le modèle réellement chargé.
COLONNES_A_SUPPRIMER = [
    "Order_ID",
    "Customer_ID",
    "Product_Name",
    "Order_Date",
    "Return_Date",
    "Refund_Amount_DA",
    "Customer_Satisfaction",
    'Return_Shipping_Paid_By',
    "Label_Source",
    "Shop_Name",
    "Shipping_Method",
]

# Features catégorielles attendues après retrait des colonnes ci-dessus.
# Valeur indicative : la source de vérité à l'inférence est
# `ohe.feature_names_in_`, et à l'entraînement l'auto-détection des colonnes non
# numériques (cf. pipeline.encoder_features). Cette liste sert de référence de
# lecture et de garde-fou de test.
COLONNES_CATEGORIEL = [
    "Customer_Gender",
    "Customer_Wilaya",
    "Product_Category",
    "Payment_Method",
    "Return_Reason",
    "reason_x_policy",
]

# ═══════════════════════════════════════════════════════════════
#  ORIGINE DU LABEL — prédiction ≠ vérité terrain
# ═══════════════════════════════════════════════════════════════
# `Resolution` est la colonne cible du modèle. Sans qualification de son
# origine, une résolution écrite par le modèle lui-même redevient un label
# d'entraînement au cycle suivant : le modèle devient sa propre vérité terrain.
#
#   "human"       → un vendeur ou un admin a tranché (PATCH /api/claims/[id]).
#                   Seule vérité terrain observée.
#   "policy_rule" → refus déterministe par la politique de retour du vendeur.
#                   Règle métier écrite par un humain, pas une sortie de modèle :
#                   utilisable comme label.
#   "model"       → recommandation du modèle, éventuellement appliquée
#                   automatiquement. JAMAIS un label d'entraînement.
#   "synthetic"   → dataset simulé (ecommerce_returns_smart_dataset.csv). Ses
#                   labels ne viennent pas du modèle Flowmerce : ils sont
#                   utilisables, à condition que l'origine soit déclarée
#                   explicitement au lancement du pipeline.
LABEL_SOURCE_HUMAN       = "human"
LABEL_SOURCE_POLICY_RULE = "policy_rule"
LABEL_SOURCE_SYNTHETIC   = "synthetic"
LABEL_SOURCE_MODEL       = "model"

LABEL_SOURCES = (
    LABEL_SOURCE_HUMAN,
    LABEL_SOURCE_POLICY_RULE,
    LABEL_SOURCE_SYNTHETIC,
    LABEL_SOURCE_MODEL,
)

# Les seules origines admises comme vérité terrain supervisée à l'entraînement.
# `model` en est exclue par construction : c'est tout l'objet de la colonne.
LABEL_SOURCES_VERITE_TERRAIN = (
    LABEL_SOURCE_HUMAN,
    LABEL_SOURCE_POLICY_RULE,
    LABEL_SOURCE_SYNTHETIC,
)

# Schéma du CSV alimenté par /save_claim. Doit rester aligné sur les champs
# de ReclamationInput (api/server.py).
CSV_COLUMNS = [
    "Order_ID", "Customer_ID", "Customer_Age", "Customer_Gender",
    "Customer_Wilaya", "Customer_Past_Returns", "Shop_Name",
    "Product_Category", "Product_Name", "Product_Price_DA",
    "Order_Quantity", "Total_Amount_DA", "Payment_Method",
    "Shipping_Method", "Shipping_Cost_DA", "Order_Date", "Return_Date",
    "Days_to_Return", "Shop_Return_Window_Days", "Within_Return_Policy",
    "Return_Reason", "Resolution", "Label_Source", "Fraud_Score",
    "Is_Suspicious", "Customer_Satisfaction",
]

# Colonne portant l'origine du label dans le dataset.
COLONNE_LABEL_SOURCE = "Label_Source"


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

USE_HF_MODELS= True
