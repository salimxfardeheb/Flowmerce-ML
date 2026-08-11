"""
C-01 — Authentification de l'API ML : fail-closed.

Le défaut corrigé : `INTERNAL_KEY` valait `None` quand la variable
d'environnement était absente, `APIKeyHeader(auto_error=False)` fournissait
`None` quand l'en-tête l'était aussi, et `None != None` étant faux, la garde
laissait passer. Une API déployée sans secret servait `/predict` et
`/save_claim` à tout le monde.

Matrice couverte ici :

    secret configuré + clé correcte   → 200/201
    secret configuré + clé incorrecte → 403
    secret configuré + clé absente    → 401
    secret NON configuré + clé absente→ 503   (et surtout : jamais 2xx)
    secret NON configuré + clé fournie→ 503   (et surtout : jamais 2xx)
"""

import pytest
from fastapi.testclient import TestClient

from conftest import CLE_TEST


ENDPOINTS_PROTEGES = [
    ("POST", "/predict"),
    ("POST", "/save_claim"),
    ("GET", "/feature-contract"),
]


def appeler(client, methode, chemin, entetes=None, payload=None):
    if methode == "GET":
        return client.get(chemin, headers=entetes or {})
    return client.post(chemin, json=payload or {}, headers=entetes or {})


# ═══════════════════════════════════════════════════════════════
#  Secret configuré
# ═══════════════════════════════════════════════════════════════
def test_secret_configure_cle_correcte(client, entetes, reclamation):
    """La bonne clé passe."""
    reponse = client.post("/save_claim", json=reclamation, headers=entetes)
    assert reponse.status_code == 201, reponse.text


@pytest.mark.parametrize("methode,chemin", ENDPOINTS_PROTEGES)
def test_secret_configure_cle_incorrecte(client, methode, chemin, reclamation):
    """Une clé erronée est rejetée en 403 sur tous les endpoints sensibles."""
    reponse = appeler(
        client, methode, chemin,
        entetes={"X-Internal-Key": "mauvaise-cle"},
        payload=reclamation,
    )
    assert reponse.status_code == 403, reponse.text


@pytest.mark.parametrize("methode,chemin", ENDPOINTS_PROTEGES)
def test_secret_configure_cle_absente(client, methode, chemin, reclamation):
    """Aucun en-tête = 401, jamais un accès."""
    reponse = appeler(client, methode, chemin, entetes={}, payload=reclamation)
    assert reponse.status_code == 401, reponse.text


def test_secret_configure_cle_vide(client, reclamation):
    """Un en-tête présent mais vide ne vaut pas authentification."""
    reponse = client.post(
        "/save_claim", json=reclamation, headers={"X-Internal-Key": ""}
    )
    assert reponse.status_code == 401, reponse.text


def test_comparaison_insensible_a_la_casse_refusee(client, reclamation):
    """La comparaison est exacte : une variante de casse ne passe pas."""
    reponse = client.post(
        "/save_claim", json=reclamation, headers={"X-Internal-Key": CLE_TEST.upper()}
    )
    assert reponse.status_code == 403, reponse.text


# ═══════════════════════════════════════════════════════════════
#  Secret NON configuré — le cœur de C-01
# ═══════════════════════════════════════════════════════════════
@pytest.fixture
def client_sans_secret(server_module, csv_reclamations, monkeypatch):
    """API dont la variable INTERNAL_API_KEY n'est pas définie."""
    monkeypatch.setattr(server_module, "RAW_DATASET_REAL", str(csv_reclamations))
    monkeypatch.setattr(server_module, "INTERNAL_KEY", None)
    return TestClient(server_module.app)


@pytest.mark.parametrize("methode,chemin", ENDPOINTS_PROTEGES)
def test_secret_absent_cle_absente(client_sans_secret, methode, chemin, reclamation):
    """
    Le cas historique du fail-open : aucune variable, aucun en-tête.
    Doit répondre 503 — et surtout jamais un 2xx.
    """
    reponse = appeler(client_sans_secret, methode, chemin, entetes={}, payload=reclamation)
    assert reponse.status_code == 503, reponse.text
    assert reponse.status_code < 200 or reponse.status_code >= 300


@pytest.mark.parametrize("methode,chemin", ENDPOINTS_PROTEGES)
def test_secret_absent_cle_fournie(client_sans_secret, methode, chemin, reclamation):
    """
    Aucune clé n'est valable quand le serveur n'en a pas : personne ne peut
    « deviner » le secret vide.
    """
    reponse = appeler(
        client_sans_secret, methode, chemin,
        entetes={"X-Internal-Key": "nimporte-quoi"},
        payload=reclamation,
    )
    assert reponse.status_code == 503, reponse.text


def test_secret_absent_naccepte_pas_none_litteral(client_sans_secret, reclamation):
    """`X-Internal-Key: None` ne doit pas se comparer favorablement à un secret absent."""
    reponse = client_sans_secret.post(
        "/save_claim", json=reclamation, headers={"X-Internal-Key": "None"}
    )
    assert reponse.status_code == 503, reponse.text


def test_secret_blanc_vaut_non_configure(server_module, csv_reclamations, monkeypatch, reclamation):
    """Un secret réduit à des espaces est traité comme absent, pas comme valide."""
    monkeypatch.setattr(server_module, "RAW_DATASET_REAL", str(csv_reclamations))
    monkeypatch.setattr(server_module, "INTERNAL_KEY", "   ")
    client = TestClient(server_module.app)

    assert client.post("/save_claim", json=reclamation,
                       headers={"X-Internal-Key": "   "}).status_code == 503
    assert client.post("/save_claim", json=reclamation).status_code == 503


# ═══════════════════════════════════════════════════════════════
#  Démarrage — l'API ne doit pas exister sans secret
# ═══════════════════════════════════════════════════════════════
def test_demarrage_refuse_sans_secret(server_module, monkeypatch):
    """
    Le cycle de vie de l'application échoue si le secret n'est pas configuré :
    le processus ne peut pas se mettre à servir.
    """
    monkeypatch.setattr(server_module, "INTERNAL_KEY", None)

    with pytest.raises(RuntimeError, match="INTERNAL_API_KEY"):
        with TestClient(server_module.app):
            pass


def test_health_expose_letat_de_configuration(client):
    """/health dit si l'authentification est configurée, sans révéler le secret."""
    corps = client.get("/health").json()
    assert corps["auth_configured"] is True
    assert CLE_TEST not in str(corps)


def test_health_reste_public(client_sans_secret):
    """La sonde de santé reste joignable même quand l'API refuse tout le reste."""
    assert client_sans_secret.get("/health").status_code == 200
    assert client_sans_secret.get("/health").json()["auth_configured"] is False
