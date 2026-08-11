"""
C-02 / C-03 — Contrat de features : source de vérité unique, dérive détectable.

Le contrat n'est pas une seconde déclaration du schéma : il est calculé depuis
les artefacts entraînés. Ces tests garantissent que cette propriété tient, et
qu'une divergence — dans un sens ou dans l'autre — est détectée au lieu d'être
absorbée par `handle_unknown="ignore"`.
"""

import copy
import json
import os

import pytest

from config import FEATURE_CONTRACT, MODELS_DIR, ENTETE_VERSION_CONTRAT
from src.feature_contract import (
    calculer_version,
    charger_contrat,
    comparer_contrats,
    contrat_depuis_artefacts,
    inspecter_categories,
)

ARTEFACTS_REELS = all(
    os.path.isfile(os.path.join(MODELS_DIR, f))
    for f in ("ohe_encoder.joblib", "scaler.joblib", "train_columns.joblib")
)


# ═══════════════════════════════════════════════════════════════
#  Le contrat versionné décrit bien les artefacts servis
# ═══════════════════════════════════════════════════════════════
def test_contrat_versionne_present(contrat):
    assert contrat is not None, f"{FEATURE_CONTRACT} absent"
    assert contrat["contract_version"]
    assert contrat["categorical_features"]
    assert contrat["numeric_features"]


@pytest.mark.skipif(not ARTEFACTS_REELS, reason="artefacts entraînés absents du dépôt")
def test_contrat_versionne_correspond_aux_artefacts(contrat):
    """
    LE test anti-dérive côté ML : régénérer le contrat depuis les artefacts
    réels doit redonner exactement le fichier versionné. Un réentraînement qui
    change un vocabulaire sans régénérer le contrat fait échouer ce test.
    """
    import joblib

    reel = contrat_depuis_artefacts(
        joblib.load(os.path.join(MODELS_DIR, "ohe_encoder.joblib")),
        joblib.load(os.path.join(MODELS_DIR, "scaler.joblib")),
        joblib.load(os.path.join(MODELS_DIR, "train_columns.joblib")),
    )

    ecarts = comparer_contrats(reel, contrat)
    assert not ecarts, (
        "Le contrat versionné ne correspond plus aux artefacts :\n  - "
        + "\n  - ".join(ecarts)
        + "\nRelancer : python scripts/build_feature_contract.py"
    )


def test_version_change_si_le_vocabulaire_change(contrat):
    """La version est une empreinte du contenu : un ajout de catégorie la change."""
    modifie = copy.deepcopy(contrat)
    modifie["categorical_features"]["Product_Category"]["categories"].append("Bijoux")

    assert calculer_version(modifie) != calculer_version(contrat)


def test_version_stable_si_rien_ne_change(contrat):
    assert calculer_version(contrat) == contrat["contract_version"]


def test_comparaison_signale_une_categorie_manquante(contrat):
    ampute = copy.deepcopy(contrat)
    ampute["categorical_features"]["Payment_Method"]["categories"].remove("CCP")

    ecarts = comparer_contrats(contrat, ampute)
    assert any("Payment_Method" in e for e in ecarts)


# ═══════════════════════════════════════════════════════════════
#  Détection des valeurs hors vocabulaire
# ═══════════════════════════════════════════════════════════════
def test_valeurs_du_contrat_sont_reconnues(contrat, prediction_payload):
    inspection = inspecter_categories(contrat, prediction_payload)

    assert inspection["unknown"] == {}
    assert inspection["alert"] == []
    assert inspection["coverage"] == 1.0


def test_ancien_vocabulaire_web_app_est_detecte(contrat):
    """
    Le vocabulaire que la web app émettait avant la correction (C-02) est
    intégralement hors du modèle. Ce test fige la détection : ces valeurs ne
    doivent plus jamais passer inaperçues.
    """
    ancien = {
        "Customer_Gender":  "Unknown",
        "Customer_Wilaya":  "Unknown",
        "Shop_Name":        "ia-store",
        "Product_Category": "Clothing",
        "Payment_Method":   "Unknown",
        "Shipping_Method":  "Standard",
        "Return_Reason":    "Mauvaise taille",
    }

    inspection = inspecter_categories(contrat, ancien)

    assert set(inspection["unknown"]) == {
        "Customer_Gender", "Customer_Wilaya", "Shop_Name",
        "Product_Category", "Payment_Method", "Shipping_Method",
    }
    # Seul Return_Reason était reconnu — 1 groupe sur 7.
    assert inspection["known"] == ["Return_Reason"]
    # Shop_Name et Shipping_Method sont des divergences documentées (features en
    # retrait) ; les trois autres restent des anomalies à corriger.
    assert set(inspection["expected"]) == FEATURES_EN_RETRAIT
    assert set(inspection["alert"]) == {
        "Customer_Gender", "Customer_Wilaya", "Payment_Method", "Product_Category",
    }


# Features en cours de retrait du modèle : leur couverture n'a plus d'enjeu, et
# alerter dessus produirait un bruit permanent que personne ne lirait.
#   Shop_Name       cardinalité non bornée — aucun vendeur réel dans les 80
#                   boutiques simulées apprises
#   Shipping_Method vocabulaire vivant, apport marginal ; la web app transmet
#                   désormais le transporteur tel quel
FEATURES_EN_RETRAIT = {"Shop_Name", "Shipping_Method"}


def test_politique_par_feature_est_explicite(contrat):
    """
    Chaque feature déclare comment traiter une valeur hors vocabulaire. Les
    features en cours de retrait sont `expected` — signalées, jamais alertées ;
    toutes les autres sont `alert`.
    """
    for nom, spec in contrat["categorical_features"].items():
        attendu = "expected" if nom in FEATURES_EN_RETRAIT else "alert"
        assert spec["unknown_policy"] == attendu, nom


def test_features_en_retrait_ne_declenchent_pas_dalerte(contrat):
    inspection = inspecter_categories(
        contrat, {"Shop_Name": "ia-store", "Shipping_Method": "Standard"}
    )

    assert set(inspection["unknown"]) == FEATURES_EN_RETRAIT
    assert inspection["alert"] == []
    assert set(inspection["expected"]) == FEATURES_EN_RETRAIT


# ═══════════════════════════════════════════════════════════════
#  Exposition par l'API
# ═══════════════════════════════════════════════════════════════
def test_endpoint_feature_contract(client, entetes, contrat):
    reponse = client.get("/feature-contract", headers=entetes)

    assert reponse.status_code == 200
    assert reponse.json()["contract_version"] == contrat["contract_version"]


def test_health_annonce_la_version(client, contrat):
    assert client.get("/health").json()["feature_contract_version"] == contrat["contract_version"]


def test_version_incompatible_est_refusee(client, entetes, prediction_payload):
    """
    Un appelant construit sur un autre vocabulaire reçoit un refus explicite —
    et non une prédiction rendue sur des features qu'il ne remplit pas comme
    le modèle les attend.
    """
    reponse = client.post(
        "/predict",
        json=prediction_payload,
        headers={**entetes, ENTETE_VERSION_CONTRAT: "version-obsolete"},
    )

    assert reponse.status_code == 409, reponse.text
    assert "contrat" in reponse.json()["detail"].lower()


def test_version_identique_est_acceptee(client, entetes, prediction_payload, contrat):
    reponse = client.post(
        "/predict",
        json=prediction_payload,
        headers={**entetes, ENTETE_VERSION_CONTRAT: contrat["contract_version"]},
    )

    assert reponse.status_code == 200, reponse.text
