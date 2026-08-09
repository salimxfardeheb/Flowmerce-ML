"""
Fixtures partagées des tests d'API.

`api/server.py` charge ses artefacts (modèle, OHE, scaler…) au moment de
l'import, depuis Hugging Face. Les tests neutralisent ce chargement avant
d'importer le module : aucun réseau, aucun token, aucun artefact local requis.
"""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CLE_TEST = "cle-interne-de-test"


class _FauxModele:
    """Modèle minimal — suffisant pour les tests qui n'évaluent pas la qualité."""

    def predict(self, X):
        return [0] * len(X)

    def predict_proba(self, X):
        return [[0.7, 0.2, 0.1]] * len(X)


def _faux_artefact(chemin, *args, **kwargs):
    nom = os.path.basename(str(chemin))
    return {
        "model_resolution.joblib": _FauxModele(),
        "ohe_encoder.joblib":      object(),
        "scaler.joblib":           object(),
        "train_columns.joblib":    [],
        "training_params.joblib":  {"seuil_risque": 3.0},
    }[nom]


@pytest.fixture(scope="session")
def server_module():
    """Importe api.server une seule fois, avec le chargement d'artefacts neutralisé."""
    import huggingface_hub
    import joblib

    hf_origine     = huggingface_hub.hf_hub_download
    joblib_origine = joblib.load

    # Renvoie le nom de fichier tel quel : _faux_artefact le résout ensuite.
    huggingface_hub.hf_hub_download = lambda filename, **kwargs: filename
    joblib.load = _faux_artefact

    try:
        from api import server
    finally:
        huggingface_hub.hf_hub_download = hf_origine
        joblib.load = joblib_origine

    return server


@pytest.fixture
def csv_reclamations(tmp_path):
    """Chemin d'un CSV de réclamations isolé, propre à chaque test."""
    return tmp_path / "reclamations.csv"


@pytest.fixture
def client(server_module, csv_reclamations, monkeypatch):
    """TestClient pointant sur un CSV temporaire et une clé interne connue."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server_module, "RAW_DATASET_REAL", str(csv_reclamations))
    monkeypatch.setattr(server_module, "INTERNAL_KEY", CLE_TEST)

    return TestClient(server_module.app)


@pytest.fixture
def entetes():
    return {"X-Internal-Key": CLE_TEST}


@pytest.fixture
def reclamation():
    """
    Payload tel que l'envoie désormais la web app : sans
    `Return_Shipping_Paid_By` ni `Refund_Amount_DA`.
    """
    return {
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
        "Fraud_Score":             5.0,
        "Is_Suspicious":           0,
        "Customer_Satisfaction":   3,
    }


def lire_csv(chemin):
    """Retourne (colonnes, lignes) d'un CSV écrit par /save_claim."""
    with open(chemin, newline="", encoding="utf-8") as f:
        lecteur = csv.DictReader(f)
        return lecteur.fieldnames, list(lecteur)
