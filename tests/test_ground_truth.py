"""
C-04 — Prédiction ≠ vérité terrain.

Trois notions distinctes, qui étaient confondues dans une seule colonne :

    prediction              recommandation du modèle
    final business decision ce qui est appliqué à la réclamation
    ground truth            ce qu'un humain (ou une règle métier) a réellement
                            décidé — la seule chose apprenable

`Label_Source` porte cette distinction jusque dans le dataset, et le pipeline
d'entraînement refuse tout ce qui n'est pas une vérité terrain.
"""

import pandas as pd
import pytest

from config import (
    LABEL_SOURCE_HUMAN,
    LABEL_SOURCE_MODEL,
    LABEL_SOURCE_POLICY_RULE,
    LABEL_SOURCE_SYNTHETIC,
    LABEL_SOURCES_VERITE_TERRAIN,
    COLONNE_LABEL_SOURCE,
    COLONNES_A_SUPPRIMER,
)
from src.pipeline import DatasetNonSupervise, filtrer_verite_terrain
from conftest import lire_csv


# ═══════════════════════════════════════════════════════════════
#  Le contrat /save_claim porte l'origine du label
# ═══════════════════════════════════════════════════════════════
def test_label_source_est_obligatoire(client, entetes, reclamation):
    """Une résolution sans provenance ne peut pas entrer dans le dataset."""
    sans = {k: v for k, v in reclamation.items() if k != "Label_Source"}

    assert client.post("/save_claim", json=sans, headers=entetes).status_code == 422


def test_label_source_invalide_est_refuse(client, entetes, reclamation):
    reponse = client.post(
        "/save_claim", json={**reclamation, "Label_Source": "devine"}, headers=entetes
    )
    assert reponse.status_code == 422


@pytest.mark.parametrize(
    "origine", [LABEL_SOURCE_HUMAN, LABEL_SOURCE_POLICY_RULE, LABEL_SOURCE_MODEL]
)
def test_origine_est_persistee_dans_le_dataset(
    client, entetes, reclamation, csv_reclamations, origine
):
    """Toutes les origines sont acceptées et tracées — le tri se fait à l'entraînement."""
    client.post(
        "/save_claim",
        json={**reclamation, "Label_Source": origine},
        headers=entetes,
    )

    colonnes, lignes = lire_csv(csv_reclamations)
    assert COLONNE_LABEL_SOURCE in colonnes
    assert lignes[0][COLONNE_LABEL_SOURCE] == origine


def test_dataset_herite_sans_colonne_est_refuse(client, entetes, reclamation, csv_reclamations):
    """
    Écrire dans un fichier antérieur au contrat ferait disparaître l'origine
    (`extrasaction="ignore"`) : la ligne deviendrait indistinguable d'un label
    humain. Refus explicite.
    """
    import csv as csv_module

    with open(csv_reclamations, "w", newline="", encoding="utf-8") as f:
        csv_module.writer(f).writerow(["Order_ID", "Resolution", "Fraud_Score"])

    reponse = client.post("/save_claim", json=reclamation, headers=entetes)

    assert reponse.status_code == 409
    assert COLONNE_LABEL_SOURCE in reponse.json()["detail"]


# ═══════════════════════════════════════════════════════════════
#  Idempotence de la collecte
# ═══════════════════════════════════════════════════════════════
def test_meme_order_id_nest_insere_quune_fois(client, entetes, reclamation, csv_reclamations):
    premiere = client.post("/save_claim", json=reclamation, headers=entetes)
    seconde  = client.post("/save_claim", json=reclamation, headers=entetes)

    assert premiere.status_code == 201
    assert seconde.status_code == 200
    assert seconde.json()["status"] == "duplicate"

    _, lignes = lire_csv(csv_reclamations)
    assert len(lignes) == 1


# ═══════════════════════════════════════════════════════════════
#  Le pipeline d'entraînement refuse les labels du modèle
# ═══════════════════════════════════════════════════════════════
def _jeu(origines):
    return pd.DataFrame(
        {
            "Order_ID": [f"ORD-{i}" for i in range(len(origines))],
            "Resolution": ["Exchange"] * len(origines),
            COLONNE_LABEL_SOURCE: origines,
        }
    )


def test_label_humain_est_conserve():
    df = filtrer_verite_terrain(_jeu([LABEL_SOURCE_HUMAN]))
    assert len(df) == 1


def test_label_regle_metier_est_conserve():
    df = filtrer_verite_terrain(_jeu([LABEL_SOURCE_POLICY_RULE]))
    assert len(df) == 1


def test_label_modele_est_exclu():
    """Le cœur de C-04 : une prédiction ne redevient jamais un label."""
    df = filtrer_verite_terrain(
        _jeu([LABEL_SOURCE_HUMAN, LABEL_SOURCE_MODEL, LABEL_SOURCE_MODEL])
    )

    assert len(df) == 1
    assert df[COLONNE_LABEL_SOURCE].tolist() == [LABEL_SOURCE_HUMAN]


def test_dataset_entierement_genere_par_le_modele_interrompt_lentrainement():
    """
    Sans aucun label humain, le pipeline ne doit pas prétendre disposer d'un
    dataset supervisé : il échoue avec un message actionnable.
    """
    with pytest.raises(DatasetNonSupervise, match="Aucune ligne de vérité terrain"):
        filtrer_verite_terrain(_jeu([LABEL_SOURCE_MODEL, LABEL_SOURCE_MODEL]))


def test_dataset_sans_colonne_de_provenance_est_refuse():
    df = pd.DataFrame({"Order_ID": ["ORD-1"], "Resolution": ["Exchange"]})

    with pytest.raises(DatasetNonSupervise, match="absente du dataset"):
        filtrer_verite_terrain(df)


def test_origine_declarable_au_lancement_pour_le_dataset_simule():
    """
    Le dataset simulé n'a pas de colonne de provenance ; ses labels ne viennent
    pas du modèle Flowmerce. L'origine se déclare explicitement — jamais par
    défaut silencieux.
    """
    df = pd.DataFrame({"Order_ID": ["ORD-1"], "Resolution": ["Exchange"]})

    resultat = filtrer_verite_terrain(df, origine_declaree=LABEL_SOURCE_SYNTHETIC)

    assert len(resultat) == 1
    assert resultat[COLONNE_LABEL_SOURCE].iloc[0] == LABEL_SOURCE_SYNTHETIC


def test_origine_declaree_model_est_refusee():
    df = pd.DataFrame({"Order_ID": ["ORD-1"], "Resolution": ["Exchange"]})

    with pytest.raises(DatasetNonSupervise, match="sorties de modèle"):
        filtrer_verite_terrain(df, origine_declaree=LABEL_SOURCE_MODEL)


def test_model_nest_jamais_une_source_de_verite_terrain():
    assert LABEL_SOURCE_MODEL not in LABEL_SOURCES_VERITE_TERRAIN


def test_label_source_nest_pas_une_feature():
    """La colonne qualifie la cible ; elle ne doit pas entrer dans le modèle."""
    assert COLONNE_LABEL_SOURCE in COLONNES_A_SUPPRIMER
