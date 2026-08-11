"""
C-03 / C-06 — Contrat de /predict.

Avant correction : `ReturnRequest` héritait du `extra="ignore"` par défaut de
Pydantic. `Customer_ID` et `Is_Suspicious` — pourtant envoyés par la web app —
étaient supprimés sans erreur ni log, puis `Is_Suspicious` était recalculé par
le serveur avec une troisième définition (`Fraud_Score >= 60`), différente de
celle apprise et de celle écrite dans le dataset.
"""

import pytest


# ═══════════════════════════════════════════════════════════════
#  Champs autrefois supprimés en silence
# ═══════════════════════════════════════════════════════════════
def test_is_suspicious_est_declare(server_module):
    assert "Is_Suspicious" in server_module.ReturnRequest.model_fields


def test_customer_id_est_declare(server_module):
    assert "Customer_ID" in server_module.ReturnRequest.model_fields


def test_aucun_champ_du_payload_web_nest_supprime(server_module, prediction_payload):
    """Tout ce que la web app envoie est retenu par le modèle Pydantic."""
    retenus = set(
        server_module.ReturnRequest(**prediction_payload).model_dump(exclude_unset=True)
    )
    assert set(prediction_payload) - retenus == set()


def test_schema_refuse_les_champs_hors_contrat(server_module):
    assert server_module.ReturnRequest.model_config.get("extra") == "forbid"


def test_champ_inconnu_provoque_un_422(client, entetes, prediction_payload):
    """
    Une divergence de schéma devient bruyante : un champ que le serveur ne
    connaît pas fait échouer la requête, au lieu de disparaître.
    """
    reponse = client.post(
        "/predict",
        json={**prediction_payload, "Customer_Satisfaction": 4},
        headers=entetes,
    )
    assert reponse.status_code == 422, reponse.text


def test_is_suspicious_est_obligatoire(client, entetes, prediction_payload):
    sans = {k: v for k, v in prediction_payload.items() if k != "Is_Suspicious"}
    assert client.post("/predict", json=sans, headers=entetes).status_code == 422


# ═══════════════════════════════════════════════════════════════
#  Is_Suspicious n'est plus réécrit par le serveur
# ═══════════════════════════════════════════════════════════════
def test_is_suspicious_de_lappelant_est_conserve(client, entetes, prediction_payload):
    """
    Un client marqué suspect par le seuil vendeur (`Customer_Past_Returns >=
    fraudReturnThreshold`) le reste, même avec un Fraud_Score bas — le serveur
    ne le recalcule plus en `Fraud_Score >= 60`.
    """
    reponse = client.post(
        "/predict",
        json={**prediction_payload, "Is_Suspicious": 1, "Fraud_Score": 5.0},
        headers=entetes,
    )

    assert reponse.status_code == 200, reponse.text
    assert reponse.json()["risk_flag"]["is_suspicious"] is True


def test_is_suspicious_zero_reste_zero_malgre_un_fraud_score_eleve(
    client, entetes, prediction_payload
):
    """Le pendant : un Fraud_Score de 90 ne force plus is_suspicious à 1."""
    reponse = client.post(
        "/predict",
        json={**prediction_payload, "Is_Suspicious": 0, "Fraud_Score": 90.0},
        headers=entetes,
    )

    assert reponse.status_code == 200, reponse.text
    assert reponse.json()["risk_flag"]["is_suspicious"] is False


# ═══════════════════════════════════════════════════════════════
#  Remontée de l'état du contrat dans la réponse
# ═══════════════════════════════════════════════════════════════
def test_reponse_porte_letat_du_contrat(client, entetes, prediction_payload, contrat):
    corps = client.post("/predict", json=prediction_payload, headers=entetes).json()

    assert corps["contract"]["version"] == contrat["contract_version"]
    assert corps["contract"]["degraded"] is False
    assert corps["contract"]["unknown_categories"] == {}
    assert corps["contract"]["categorical_coverage"] == 1.0


def test_valeur_hors_vocabulaire_est_signalee(client, entetes, prediction_payload):
    """
    Une catégorie inconnue produit un vecteur one-hot nul. La prédiction est
    rendue, mais l'appelant est informé — c'est ce que l'audit exigeait :
    « journaliser/alerter sur tout taux de catégories inconnues non nul ».
    """
    corps = client.post(
        "/predict",
        json={**prediction_payload, "Product_Category": "Clothing"},
        headers=entetes,
    ).json()

    assert corps["contract"]["degraded"] is True
    assert corps["contract"]["unknown_categories"] == {"Product_Category": "Clothing"}
    assert corps["contract"]["alert_features"] == ["Product_Category"]
    assert corps["contract"]["categorical_coverage"] < 1.0


def test_shop_name_inconnu_nest_pas_une_alerte(client, entetes, prediction_payload):
    """Divergence documentée : signalée, mais pas remontée comme anomalie."""
    corps = client.post(
        "/predict",
        json={**prediction_payload, "Shop_Name": "ia-store"},
        headers=entetes,
    ).json()

    assert corps["contract"]["degraded"] is True
    assert corps["contract"]["alert_features"] == []
    assert corps["contract"]["expected_unknown"] == ["Shop_Name"]


def test_alerte_journalisee_sur_categorie_inconnue(client, entetes, prediction_payload, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="flowmerce.api"):
        client.post(
            "/predict",
            json={**prediction_payload, "Payment_Method": "Unknown"},
            headers=entetes,
        )

    assert any("categories_inconnues" in m for m in caplog.messages)
