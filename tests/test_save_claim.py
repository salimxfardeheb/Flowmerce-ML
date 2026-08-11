"""
Contrat de /save_claim après le retrait de `Return_Shipping_Paid_By`
et `Refund_Amount_DA`.
"""

import csv

from conftest import lire_csv

CHAMPS_RETIRES = ["Return_Shipping_Paid_By", "Refund_Amount_DA"]


# ═══════════════════════════════════════════════════════════════
#  Nouveau contrat — payload sans les deux champs
# ═══════════════════════════════════════════════════════════════
def test_payload_sans_champs_retires_accepte(client, entetes, reclamation):
    """Le payload de la web app migrée est accepté."""
    reponse = client.post("/save_claim", json=reclamation, headers=entetes)

    assert reponse.status_code == 201, reponse.text
    assert reponse.json()["order_id"] == reclamation["Order_ID"]


def test_csv_ne_contient_plus_les_colonnes_retirees(
    client, entetes, reclamation, csv_reclamations
):
    """Le dataset produit ne porte plus les deux colonnes."""
    client.post("/save_claim", json=reclamation, headers=entetes)

    colonnes, lignes = lire_csv(csv_reclamations)

    for champ in CHAMPS_RETIRES:
        assert champ not in colonnes

    assert len(lignes) == 1
    assert lignes[0]["Order_ID"] == reclamation["Order_ID"]
    assert lignes[0]["Resolution"] == "Exchange"


def test_insertions_successives_sans_reecriture_entete(
    client, entetes, reclamation, csv_reclamations
):
    """L'en-tête n'est écrit qu'une fois, quel que soit le nombre d'insertions."""
    for i in range(3):
        client.post(
            "/save_claim",
            json={**reclamation, "Order_ID": f"ORD-{i}"},
            headers=entetes,
        )

    colonnes, lignes = lire_csv(csv_reclamations)

    assert len(lignes) == 3
    assert [l["Order_ID"] for l in lignes] == ["ORD-0", "ORD-1", "ORD-2"]
    assert "Order_ID" not in [l["Order_ID"] for l in lignes]  # pas d'en-tête dupliqué
    for champ in CHAMPS_RETIRES:
        assert champ not in colonnes


# ═══════════════════════════════════════════════════════════════
#  Fenêtre de transition — ancien client
# ═══════════════════════════════════════════════════════════════
def test_ancien_payload_avec_champs_retires_nechoue_pas(
    client, entetes, reclamation, csv_reclamations
):
    """
    Un client non migré envoie encore les deux clés : extra="ignore" doit
    répondre 201, et non 422.
    """
    ancien_payload = {
        **reclamation,
        "Return_Shipping_Paid_By": "Marchand",
        "Refund_Amount_DA":        0.0,
    }

    reponse = client.post("/save_claim", json=ancien_payload, headers=entetes)

    assert reponse.status_code == 201, reponse.text

    # …et les valeurs héritées ne sont pas persistées.
    colonnes, lignes = lire_csv(csv_reclamations)
    for champ in CHAMPS_RETIRES:
        assert champ not in colonnes
    assert "Marchand" not in lignes[0].values()


def test_csv_herite_nest_pas_decale(client, entetes, reclamation, csv_reclamations):
    """
    Sur un CSV portant l'ancien en-tête (28 colonnes, dont les deux retirées),
    l'insertion suit cet en-tête : aucune valeur ne glisse d'une colonne, les
    colonnes retirées restent vides.
    """
    entete_heritee = [
        "Order_ID", "Customer_ID", "Customer_Age", "Customer_Gender",
        "Customer_Wilaya", "Customer_Past_Returns", "Shop_Name",
        "Product_Category", "Product_Name", "Product_Price_DA",
        "Order_Quantity", "Total_Amount_DA", "Payment_Method",
        "Shipping_Method", "Shipping_Cost_DA", "Order_Date", "Return_Date",
        "Days_to_Return", "Shop_Return_Window_Days", "Within_Return_Policy",
        "Return_Reason", "Resolution", "Label_Source",
        "Return_Shipping_Paid_By", "Refund_Amount_DA", "Fraud_Score",
        "Is_Suspicious", "Customer_Satisfaction",
    ]
    with open(csv_reclamations, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(entete_heritee)

    reponse = client.post("/save_claim", json=reclamation, headers=entetes)
    assert reponse.status_code == 201, reponse.text

    colonnes, lignes = lire_csv(csv_reclamations)

    assert colonnes == entete_heritee
    ligne = lignes[0]
    # Les champs situés APRÈS les colonnes retirées restent à leur place.
    assert ligne["Fraud_Score"]           == "5.0"
    assert ligne["Is_Suspicious"]         == "0"
    assert ligne["Customer_Satisfaction"] == "3"
    # Les colonnes héritées ne sont plus alimentées.
    for champ in CHAMPS_RETIRES:
        assert ligne[champ] == ""


# ═══════════════════════════════════════════════════════════════
#  Schéma exposé
# ═══════════════════════════════════════════════════════════════
def test_openapi_nexpose_plus_les_champs_retires(client):
    """Ni /save_claim ni /predict ne déclarent les deux champs."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    for nom in ("ReclamationInput", "ReturnRequest"):
        proprietes = schemas[nom]["properties"]
        for champ in CHAMPS_RETIRES:
            assert champ not in proprietes, f"{champ} encore exposé dans {nom}"


def test_champs_retires_absents_du_schema_csv():
    """CSV_COLUMNS — source de vérité du dataset — ne les liste plus."""
    from config import CSV_COLUMNS

    for champ in CHAMPS_RETIRES:
        assert champ not in CSV_COLUMNS


def test_modele_pydantic_ignore_les_extras(server_module):
    """La fenêtre de transition repose sur extra="ignore", pas sur "forbid"."""
    assert server_module.ReclamationInput.model_config.get("extra") == "ignore"
