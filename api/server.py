import os
import csv
import logging
import secrets
import sys
from contextlib import asynccontextmanager, contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enum import Enum
from typing import Optional
from datetime import date

from fastapi import FastAPI, HTTPException, Request, Response, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd

from config import (
    MODEL_RESOLUTION,
    OHE_ENCODER,
    SCALER,
    TRAIN_COLUMNS,
    TRAINING_PARAMS,
    RESOLUTION_LABELS,
    INTERNAL_KEY,
    LONGUEUR_MIN_CLE_INTERNE,
    RAW_DATASET_REAL,
    CSV_COLUMNS,
    USE_HF_MODELS,
    HF_REPO_ID,
    HF_TOKEN,
    LABEL_SOURCE_MODEL,
    COLONNE_LABEL_SOURCE,
    ENTETE_VERSION_CONTRAT,
)

from src.preprocessing import preprocess
from src.feature_contract import (
    charger_contrat,
    comparer_contrats,
    contrat_depuis_artefacts,
    inspecter_categories,
)

logger = logging.getLogger("flowmerce.api")


# ═══════════════════════════════════════════════════════════════
#  CHARGEMENT DES ARTEFACTS — Hugging Face (statique) ou local
# ═══════════════════════════════════════════════════════════════
def charger_artefact(nom_fichier_hf, chemin_local):
    """
    Si USE_HF_MODELS est True : télécharge (ou lit le cache) depuis le repo HF.
    Sinon : lit le fichier local défini dans config.py.
    """
    if USE_HF_MODELS:
        chemin = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=nom_fichier_hf,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"[Artefact] {nom_fichier_hf} chargé depuis Hugging Face ({HF_REPO_ID})")
    else:
        chemin = chemin_local
        print(f"[Artefact] {chemin_local} chargé en local")

    return joblib.load(chemin)


model_resolution = charger_artefact("model_resolution.joblib", MODEL_RESOLUTION)
ohe              = charger_artefact("ohe_encoder.joblib",      OHE_ENCODER)
scaler           = charger_artefact("scaler.joblib",           SCALER)
train_columns    = charger_artefact("train_columns.joblib",    TRAIN_COLUMNS)
training_params  = charger_artefact("training_params.joblib",  TRAINING_PARAMS)

seuil_risque = training_params["seuil_risque"]


def lire_entete_csv(chemin):
    """
    Retourne la liste des colonnes de l'en-tête d'un CSV existant,
    ou None si le fichier n'existe pas / est vide.
    """
    if not os.path.isfile(chemin) or os.path.getsize(chemin) == 0:
        return None

    with open(chemin, mode="r", newline="", encoding="utf-8") as f:
        entete = next(csv.reader(f), None)

    return entete or None


def order_id_deja_present(chemin, order_id):
    """
    Vrai si le dataset porte déjà une ligne pour cet Order_ID.
    Clé d'idempotence de /save_claim : un export rejoué ne duplique rien.
    """
    if not os.path.isfile(chemin) or os.path.getsize(chemin) == 0:
        return False

    with open(chemin, newline="", encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            if (ligne.get("Order_ID") or "") == order_id:
                return True
    return False


@contextmanager
def verrou_exclusif(fichier):
    """
    Verrou exclusif sur le descripteur, quand la plateforme le permet.
    Sur un système sans `fcntl` (Windows), l'écriture reste possible : le verrou
    est une protection contre l'entrelacement, pas une condition d'écriture.
    """
    try:
        import fcntl
    except ImportError:
        yield
        return

    fcntl.flock(fichier.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(fichier.fileno(), fcntl.LOCK_UN)


# ═══════════════════════════════════════════════════════════════
#  AUTHENTIFICATION — X-Internal-Key  (fail-closed)
#
#  Trois situations, trois réponses distinctes :
#    • secret non configuré  → 503, AUCUNE requête n'est servie. L'absence de
#      configuration ne doit jamais ouvrir l'API (`None == None` était une
#      authentification valide dans la version précédente).
#    • en-tête absent        → 401
#    • en-tête erroné        → 403, comparaison à temps constant.
# ═══════════════════════════════════════════════════════════════
api_key_header = APIKeyHeader(name="X-Internal-Key", auto_error=False)


def cle_interne_configuree() -> Optional[str]:
    """
    Secret attendu, ou None s'il n'est pas configuré.

    Relit la variable de module à chaque appel (et non une copie figée) afin que
    la garde reste vraie même si la configuration change à chaud.
    """
    cle = INTERNAL_KEY
    if not isinstance(cle, str):
        return None
    cle = cle.strip()
    return cle or None


def verify_internal_key(api_key: Optional[str] = Security(api_key_header)):
    attendue = cle_interne_configuree()

    if attendue is None:
        logger.error(
            "auth.secret_absent — INTERNAL_API_KEY n'est pas configuré : "
            "toutes les requêtes authentifiées sont rejetées."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API mal configurée : secret interne absent. Aucune requête n'est servie.",
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé interne manquante (en-tête X-Internal-Key).",
        )

    # compare_digest : pas de fuite de temps sur les préfixes communs.
    if not secrets.compare_digest(api_key, attendue):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé interne invalide.",
        )

    return api_key


# ═══════════════════════════════════════════════════════════════
#  SCHÉMA — /predict
# ═══════════════════════════════════════════════════════════════
class ReturnRequest(BaseModel):
    # Identifiant client : transmis par la web app, jamais une feature du modèle
    # (`Customer_ID` est dans COLONNES_A_SUPPRIMER). Déclaré ici pour qu'il soit
    # accepté explicitement plutôt que supprimé en silence.
    Customer_ID:             Optional[str] = None

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

    # Feature du modèle à part entière (présente dans scaler.feature_names_in_)
    # et source de `fraud_x_suspicious`. Elle était auparavant supprimée en
    # silence par Pydantic puis recalculée ici en `Fraud_Score >= 60` —
    # une troisième sémantique, différente de celle apprise et de celle
    # persistée dans le dataset. C'est désormais l'appelant qui la définit,
    # avec le seuil vendeur qu'il a configuré, et le serveur ne la recalcule pas.
    Is_Suspicious:           int   = Field(ge=0, le=1)

    # `forbid` : tout champ hors contrat provoque un 422 explicite. Une
    # divergence de schéma entre la web app et l'API ML devient bruyante au lieu
    # de faire disparaître une feature sans trace.
    model_config = ConfigDict(extra="forbid", json_schema_extra={
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
            "Is_Suspicious":           0,
        }
    })


# ═══════════════════════════════════════════════════════════════
#  SCHÉMA — /save_claim
# ═══════════════════════════════════════════════════════════════
class ResolutionEnum(str, Enum):
    Exchange = "Exchange"
    Reject   = "Reject"
    Repair   = "Repair"
    Refund   = "Refund"


class LabelSourceEnum(str, Enum):
    """
    Origine de `Resolution`. Champ obligatoire : une résolution sans provenance
    ne peut pas être distinguée d'une sortie de modèle, et une sortie de modèle
    ne doit jamais devenir un label d'entraînement (C-04).
    """
    human       = "human"        # décision d'un vendeur ou d'un admin
    policy_rule = "policy_rule"  # refus déterministe par la politique de retour
    model       = "model"        # recommandation du modèle — pas une vérité terrain


class ReclamationInput(BaseModel):
    Order_ID:                 str
    Customer_ID:               str
    Customer_Age:              int   = Field(ge=0)
    Customer_Gender:           str
    Customer_Wilaya:           str
    Customer_Past_Returns:     int   = Field(ge=0)
    Shop_Name:                 str
    Product_Category:          str
    Product_Name:              str
    Product_Price_DA:          float = Field(gt=0)
    Order_Quantity:            int   = Field(ge=1)
    Total_Amount_DA:           float = Field(gt=0)
    Payment_Method:            str
    Shipping_Method:           str
    Shipping_Cost_DA:          float = Field(ge=0)
    Order_Date:                 date
    Return_Date:                 date
    Days_to_Return:              int   = Field(ge=0)
    Shop_Return_Window_Days:     int   = Field(gt=0)
    Within_Return_Policy:        int   = Field(ge=0, le=1)
    Return_Reason:               str
    Resolution:                   ResolutionEnum
    Label_Source:                 LabelSourceEnum
    Fraud_Score:                   float = Field(ge=0, le=100)
    Is_Suspicious:                  int  = Field(ge=0, le=1)
    Customer_Satisfaction:           Optional[int] = Field(default=None, ge=1, le=5)

    # Fenêtre de transition : `Return_Shipping_Paid_By` et `Refund_Amount_DA`
    # ne sont plus collectés par aucun point d'entrée du produit. Le client web
    # a cessé de les envoyer ; extra="ignore" garantit qu'un ancien client qui
    # les enverrait encore reçoive un 201 plutôt qu'un 422.
    # À repasser en "forbid" une fois tous les clients migrés.
    model_config = ConfigDict(extra="ignore", json_schema_extra={
        "example": {
            "Order_ID":                "cmrxl0wm9000023zzv2lryhrn",
            "Customer_ID":             "CUST-501",
            "Customer_Age":            30,
            "Customer_Gender":         "Unknown",
            "Customer_Wilaya":         "Alger",
            "Customer_Past_Returns":   0,
            "Shop_Name":               "ia-store",
            "Product_Category":        "Vetements",
            "Product_Name":            "OVERSIZE VINTAGE SHIRT",
            "Product_Price_DA":        5500.0,
            "Order_Quantity":          1,
            "Total_Amount_DA":         5500.0,
            "Payment_Method":          "Especes livraison",
            "Shipping_Method":         "Yalidine",
            "Shipping_Cost_DA":        400.0,
            "Order_Date":              "2026-07-23",
            "Return_Date":             "2026-07-31",
            "Days_to_Return":          8,
            "Shop_Return_Window_Days": 14,
            "Within_Return_Policy":    1,
            "Return_Reason":           "Mauvaise taille",
            "Resolution":              "Exchange",
            "Label_Source":            "human",
            "Fraud_Score":             5.0,
            "Is_Suspicious":           0,
            "Customer_Satisfaction":   3,
        }
    })


# ═══════════════════════════════════════════════════════════════
#  CONTRAT DE FEATURES
#
#  Chargé depuis le fichier versionné (contracts/feature_contract.json). Le
#  démarrage vérifie qu'il correspond bien aux artefacts réellement chargés :
#  c'est la garantie que ce document décrit le modèle servi, et pas un autre.
# ═══════════════════════════════════════════════════════════════
CONTRAT = charger_contrat()
VERSION_CONTRAT = (CONTRAT or {}).get("contract_version")


# ═══════════════════════════════════════════════════════════════
#  DÉMARRAGE — vérifications bloquantes (fail-closed)
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Secret interne — sans lui, l'API ne doit pas exister.
    attendue = cle_interne_configuree()
    if attendue is None:
        raise RuntimeError(
            "INTERNAL_API_KEY absent ou vide : démarrage refusé. "
            "L'API ML ne doit jamais être servie sans authentification."
        )
    if len(attendue) < LONGUEUR_MIN_CLE_INTERNE:
        logger.warning(
            "auth.secret_court — INTERNAL_API_KEY fait %d caractères "
            "(minimum recommandé : %d).",
            len(attendue),
            LONGUEUR_MIN_CLE_INTERNE,
        )

    # 2. Contrat de features — le document versionné doit décrire les artefacts
    #    effectivement chargés. Sinon les prédictions seraient rendues sur un
    #    vocabulaire différent de celui annoncé aux appelants.
    if CONTRAT is None:
        raise RuntimeError(
            "contracts/feature_contract.json absent : démarrage refusé. "
            "Générer le contrat avec `python scripts/build_feature_contract.py`."
        )

    reel = contrat_depuis_artefacts(ohe, scaler, train_columns, RESOLUTION_LABELS)
    ecarts = comparer_contrats(reel, CONTRAT)
    if ecarts:
        raise RuntimeError(
            "Contrat de features divergent des artefacts chargés — démarrage refusé :\n  - "
            + "\n  - ".join(ecarts)
            + "\nRégénérer avec `python scripts/build_feature_contract.py`, "
              "puis répercuter la copie côté web app."
        )

    logger.info("startup.ok — contrat de features %s", VERSION_CONTRAT)
    yield


# ═══════════════════════════════════════════════════════════════
#  APPLICATION FASTAPI
# ═══════════════════════════════════════════════════════════════
app = FastAPI(
    title="Flowmerce — API de Prediction des Retours",
    description="Prediction de la resolution d'un retour produit.",
    version="5.0.0",
    lifespan=lifespan,
)


def verifier_version_contrat(request: Request):
    """
    Rejette explicitement un appelant construit sur un autre vocabulaire.

    Un en-tête absent est toléré (intégrations antérieures au contrat) mais
    journalisé : la web app, elle, l'envoie systématiquement.
    """
    annoncee = request.headers.get(ENTETE_VERSION_CONTRAT)
    if annoncee is None:
        logger.warning("contract.version_absente — appelant sans %s", ENTETE_VERSION_CONTRAT)
        return None

    if VERSION_CONTRAT is not None and annoncee != VERSION_CONTRAT:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Version de contrat incompatible : appelant={annoncee}, "
                f"serveur={VERSION_CONTRAT}. Le vocabulaire des features a changé — "
                "aucune prédiction n'est rendue."
            ),
        )
    return annoncee


@app.get("/")
def root():
    return {
        "message": "Flowmerce Returns Prediction API",
        "version": "4.0.0",
        "endpoints": {
            "/predict":          "POST — Predire la resolution du retour",
            "/save_claim":       "POST — Inserer une reclamation reelle (avec resolution finale)",
            "/feature-contract": "GET  — Contrat de features servi par cette instance",
            "/health":           "GET  — Verifier l'etat de l'API",
        },
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "source_artefacts": "huggingface" if USE_HF_MODELS else "local",
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
        "feature_contract_version": VERSION_CONTRAT,
        # Ne révèle pas le secret : dit seulement s'il est configuré. Un `false`
        # ici signifie que l'API rejette 100 % des requêtes authentifiées.
        "auth_configured": cle_interne_configuree() is not None,
    }


@app.get("/feature-contract")
def feature_contract(_: str = Security(verify_internal_key)):
    """
    Contrat de features servi par cette instance — vocabulaire catégoriel appris,
    features numériques, version. C'est le document que la web app doit copier ;
    toute divergence est détectable en le comparant à sa copie locale.
    """
    if CONTRAT is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Contrat de features non chargé.",
        )
    return CONTRAT


@app.post("/predict")
def predict(
    request: ReturnRequest,
    http_request: Request,
    _: str = Security(verify_internal_key),
):
    verifier_version_contrat(http_request)

    donnees = request.model_dump()
    # `Customer_ID` est accepté au contrat mais n'est pas une feature : le retirer
    # ici évite qu'il n'entre dans le feature engineering.
    donnees.pop("Customer_ID", None)

    # Confrontation au vocabulaire appris AVANT toute prédiction. Une valeur hors
    # vocabulaire devient un vecteur one-hot nul (handle_unknown="ignore") : la
    # prédiction reste possible mais amputée. Elle n'est plus silencieuse.
    inspection = inspecter_categories(CONTRAT, donnees)
    if inspection["alert"]:
        logger.warning(
            "contract.categories_inconnues — features=%s valeurs=%s couverture=%.2f",
            inspection["alert"],
            {k: inspection["unknown"][k] for k in inspection["alert"]},
            inspection["coverage"],
        )

    try:
        row = pd.DataFrame([donnees])

        # Is_Suspicious vient de l'appelant (cf. ReturnRequest) : il porte la
        # définition métier du vendeur. Le serveur ne la réécrit plus.

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
                "client_a_risque": bool(row["Customer_Past_Returns"].iloc[0] >= seuil_risque),
            },
            # État du contrat pour CETTE prédiction : ce que le modèle a
            # réellement reconnu. `degraded` dit qu'une partie du vecteur
            # d'entrée est nulle — la prédiction est rendue, mais l'appelant
            # sait qu'elle repose sur moins d'information qu'à l'entraînement.
            "contract": {
                "version":            VERSION_CONTRAT,
                "degraded":           bool(inspection["unknown"]),
                "unknown_categories": inspection["unknown"],
                "alert_features":     inspection["alert"],
                "expected_unknown":   inspection["expected"],
                "categorical_coverage": inspection["coverage"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_claim", status_code=status.HTTP_201_CREATED)
def ajouter_reclamation(
    data: ReclamationInput,
    response: Response,
    _: str = Security(verify_internal_key),
):
    row_dict = data.model_dump()
    row_dict["Resolution"]   = row_dict["Resolution"].value      # Enum -> str
    row_dict["Label_Source"] = row_dict["Label_Source"].value

    # Un label produit par le modèle est accepté (traçabilité, analyse de
    # dérive) mais il est refusé à l'entraînement — cf. pipeline.charger_donnees.
    if row_dict["Label_Source"] == LABEL_SOURCE_MODEL:
        logger.info(
            "save_claim.label_modele — order_id=%s : ligne conservée hors vérité terrain.",
            data.Order_ID,
        )

    # Collecte : ici, contrairement à /predict, une catégorie hors vocabulaire
    # n'est PAS une anomalie — c'est le produit qui évolue, et c'est exactement
    # ce qu'il faut capter. /save_claim décrit la réalité d'une réclamation ;
    # /predict interroge un modèle figé. Une valeur nouvelle est donc acceptée
    # telle quelle et journalisée comme matière à réentraînement.
    inspection = inspecter_categories(CONTRAT, row_dict)
    if inspection["unknown"]:
        logger.info(
            "save_claim.vocabulaire_nouveau — order_id=%s valeurs=%s : "
            "hors du contrat %s, conservées pour le prochain entraînement.",
            data.Order_ID,
            inspection["unknown"],
            VERSION_CONTRAT,
        )

    try:
        # Idempotence : rejouer le même Order_ID ne duplique pas la ligne. Sans
        # cette garde, un export relancé (ou un retry réseau) faisait entrer
        # plusieurs fois la même réclamation dans le dataset.
        if order_id_deja_present(RAW_DATASET_REAL, data.Order_ID):
            response.status_code = status.HTTP_200_OK
            return {
                "status": "duplicate",
                "message": "Réclamation déjà présente dans le dataset — aucune insertion.",
                "order_id": data.Order_ID,
            }

        # Un CSV déjà présent peut porter un en-tête hérité (colonnes retirées
        # depuis, ex. Return_Shipping_Paid_By / Refund_Amount_DA). On écrit
        # alors selon SON en-tête — sinon les valeurs seraient décalées d'une
        # colonne à l'insertion. Les colonnes héritées sont laissées vides :
        # le fichier reste lisible, mais n'est plus alimenté.
        entete_existante = lire_entete_csv(RAW_DATASET_REAL)

        # …sauf pour Label_Source : `extrasaction="ignore"` la ferait disparaître
        # en silence sur un fichier antérieur au contrat, et une résolution sans
        # provenance est indistinguable d'une sortie de modèle. Refus explicite
        # plutôt que perte silencieuse (C-04).
        if entete_existante is not None and COLONNE_LABEL_SOURCE not in entete_existante:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Le dataset existant ne porte pas la colonne {COLONNE_LABEL_SOURCE} : "
                    "insertion refusée. Migrer le fichier (ajouter la colonne, "
                    "renseigner l'origine des lignes existantes) avant de reprendre "
                    "la collecte."
                ),
            )

        with open(RAW_DATASET_REAL, mode="a", newline="", encoding="utf-8") as f:
            # Écritures concurrentes : sans verrou, deux requêtes simultanées
            # produisaient des lignes entrelacées dans le fichier.
            with verrou_exclusif(f):
                writer = csv.DictWriter(
                    f,
                    fieldnames=entete_existante or CSV_COLUMNS,
                    restval="",           # colonnes héritées → vides
                    extrasaction="ignore",
                )

                if entete_existante is None:
                    writer.writeheader()

                writer.writerow(row_dict)

        return {
            "status": "ok",
            "message": "Réclamation insérée avec succès.",
            "order_id": data.Order_ID,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
