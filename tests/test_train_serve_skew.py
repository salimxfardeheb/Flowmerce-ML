"""
C-02 — Mesure empirique du skew train/serve, sur les artefacts réels du dépôt.

L'audit avait mesuré, avec ces mêmes artefacts :

    vocabulaire TRAIN    | one-hot actives : 8/157
    vocabulaire WEB APP  | one-hot actives : 2/157   ← 6 groupes sur 8 effondrés

Ces tests rejouent la mesure. Ils échouent si le vocabulaire émis par la web app
redevient incompatible avec celui appris par l'encodeur.

Ignorés si les artefacts entraînés ne sont pas présents (dépôt cloné sans les
joblib, CI sans accès Hugging Face).
"""

import json
import os

import pandas as pd
import pytest

from config import MODELS_DIR, PROJECT_ROOT
from src.feature_contract import charger_contrat
from src.preprocessing import preprocess

ARTEFACTS = ("ohe_encoder.joblib", "scaler.joblib", "train_columns.joblib", "training_params.joblib")
DISPONIBLES = all(os.path.isfile(os.path.join(MODELS_DIR, f)) for f in ARTEFACTS)

pytestmark = pytest.mark.skipif(
    not DISPONIBLES, reason="artefacts entraînés absents du dépôt"
)


@pytest.fixture(scope="module")
def artefacts():
    import joblib

    return {
        "ohe":     joblib.load(os.path.join(MODELS_DIR, "ohe_encoder.joblib")),
        "scaler":  joblib.load(os.path.join(MODELS_DIR, "scaler.joblib")),
        "columns": joblib.load(os.path.join(MODELS_DIR, "train_columns.joblib")),
        "params":  joblib.load(os.path.join(MODELS_DIR, "training_params.joblib")),
    }


def _ligne(**surcharges):
    base = {
        "Customer_Gender":         "Female",
        "Customer_Age":            34,
        "Customer_Wilaya":         "Alger",
        "Customer_Past_Returns":   1,
        "Shop_Name":               "Shop_001",
        "Product_Category":        "Vêtements",
        "Product_Price_DA":        3500.0,
        "Order_Quantity":          1,
        "Total_Amount_DA":         3500.0,
        "Payment_Method":          "Espèces livraison",
        "Shipping_Method":         "Yalidine",
        "Shipping_Cost_DA":        400.0,
        "Return_Reason":           "Mauvaise taille",
        "Days_to_Return":          4,
        "Shop_Return_Window_Days": 14,
        "Within_Return_Policy":    1,
        "Fraud_Score":             5.0,
        "Is_Suspicious":           0,
    }
    base.update(surcharges)
    return pd.DataFrame([base])


def _one_hot_actives(df, artefacts):
    X = preprocess(
        df,
        artefacts["ohe"],
        artefacts["scaler"],
        artefacts["columns"],
        seuil_risque=artefacts["params"]["seuil_risque"],
    )
    colonnes_ohe = [c for c in X.columns if c not in artefacts["scaler"].feature_names_in_]
    return int((X[colonnes_ohe].iloc[0] == 1).sum())


# ═══════════════════════════════════════════════════════════════
#  Le vocabulaire du contrat produit un vecteur complet
# ═══════════════════════════════════════════════════════════════
def test_vocabulaire_du_contrat_active_tous_les_groupes(artefacts):
    """
    8 groupes catégoriels, une modalité active par groupe : le vecteur d'entrée
    porte toute l'information attendue par le modèle.
    """
    assert _one_hot_actives(_ligne(), artefacts) == 8


def test_ancien_vocabulaire_web_app_effondrait_le_vecteur(artefacts):
    """
    Reproduction du défaut mesuré par l'audit : avec l'ancien vocabulaire de la
    web app, seules les deux colonnes dérivées de Return_Reason survivaient.
    Ce test documente l'écart et garantit que la mesure reste vraie.
    """
    ancien = _ligne(
        Customer_Gender="Unknown",
        Customer_Wilaya="Unknown",
        Shop_Name="ia-store",
        Product_Category="Clothing",
        Payment_Method="Unknown",
        Shipping_Method="Standard",
    )

    assert _one_hot_actives(ancien, artefacts) == 2


def test_vocabulaire_corrige_recupere_les_groupes_mappables(artefacts):
    """
    Après correction, la web app émet le vocabulaire du modèle. Shop_Name reste
    hors vocabulaire par construction (boutiques synthétiques) : 7 groupes sur 8.
    """
    corrige = _ligne(Shop_Name="ia-store")

    assert _one_hot_actives(corrige, artefacts) == 7


# ═══════════════════════════════════════════════════════════════
#  Cohérence avec la copie de la web app
# ═══════════════════════════════════════════════════════════════
COPIE_WEB = os.path.join(
    PROJECT_ROOT, "..", "flowmerce-web-app", "lib", "ml", "feature-contract.json"
)


# ═══════════════════════════════════════════════════════════════
#  Retrait de Shop_Name et Shipping_Method des features
# ═══════════════════════════════════════════════════════════════
def test_features_retirees_de_lentrainement():
    """
    Les deux features sont écartées avant l'encodage : le prochain modèle ne
    les apprendra pas. Elles restent collectées par /save_claim — le dataset
    doit continuer de décrire la réalité d'une réclamation.
    """
    from config import COLONNES_A_SUPPRIMER, COLONNES_CATEGORIEL, CSV_COLUMNS

    for feature in ("Shop_Name", "Shipping_Method"):
        assert feature in COLONNES_A_SUPPRIMER, f"{feature} encore encodée à l'entraînement"
        assert feature not in COLONNES_CATEGORIEL
        assert feature in CSV_COLUMNS, f"{feature} doit rester collectée"


@pytest.mark.skipif(not DISPONIBLES, reason="artefacts entraînés absents du dépôt")
def test_inference_suit_lencodeur_et_non_la_configuration(artefacts):
    """
    Les colonnes encodées sont lues sur `ohe.feature_names_in_`, pas sur
    `COLONNES_CATEGORIEL`. C'est ce qui permet au modèle actuel — qui connaît
    encore Shop_Name et Shipping_Method — de continuer à servir alors que la
    configuration d'entraînement ne les liste plus.
    """
    from config import COLONNES_CATEGORIEL

    attendues = set(artefacts["ohe"].feature_names_in_)
    assert {"Shop_Name", "Shipping_Method"} <= attendues
    assert not {"Shop_Name", "Shipping_Method"} & set(COLONNES_CATEGORIEL)

    # Le modèle déployé prédit toujours, sans changement de code.
    assert _one_hot_actives(_ligne(), artefacts) == 8


@pytest.mark.skipif(not DISPONIBLES, reason="artefacts entraînés absents du dépôt")
def test_colonne_categorielle_absente_echoue_explicitement(artefacts):
    """Une colonne attendue par l'encodeur et absente lève une erreur nommée."""
    ligne = _ligne().drop(columns=["Shop_Name"])

    with pytest.raises(ValueError, match="Shop_Name"):
        _one_hot_actives(ligne, artefacts)


@pytest.mark.skipif(
    not os.path.isfile(COPIE_WEB), reason="dépôt web app non présent à côté"
)
def test_copie_web_app_identique_au_contrat():
    """
    Détection de dérive entre les deux dépôts : la web app doit servir
    exactement le contrat que l'API ML applique.
    """
    with open(COPIE_WEB, encoding="utf-8") as f:
        copie = json.load(f)

    assert copie == charger_contrat(), (
        "La copie web app du contrat a divergé.\n"
        "→ cp contracts/feature_contract.json "
        "../flowmerce-web-app/lib/ml/feature-contract.json"
    )
