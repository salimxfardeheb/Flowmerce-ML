"""
Contrat de features — source de vérité unique du vocabulaire du modèle.

Problème traité (C-02 / C-03 de l'audit) : le vocabulaire catégoriel émis par la
web app et celui appris par l'encodeur divergeaient silencieusement.
`handle_unknown="ignore"` transformait chaque valeur inconnue en vecteur nul,
sans erreur, sans log. Six des huit groupes catégoriels étaient morts en
production.

Principe de la correction :

    artefacts entraînés  (ohe_encoder.joblib, scaler.joblib, train_columns.joblib)
              │
              ▼   contrat_depuis_artefacts()   ← DÉRIVÉ, jamais saisi à la main
        feature_contract.json   (versionné dans le dépôt, version = hash du contenu)
              │
        ┌─────┴─────┐
        ▼           ▼
   API ML       web app (copie identique, lib/ml/feature-contract.json)

Le contrat n'est donc pas une seconde déclaration du schéma susceptible de
diverger : il est *calculé* depuis les artefacts. Un réentraînement qui change
un vocabulaire change la version du contrat, et :

  - un test échoue si le fichier versionné ne correspond plus aux artefacts ;
  - un test échoue si la copie web app ne correspond plus au fichier ML ;
  - l'API refuse de démarrer si les deux divergent ;
  - l'API refuse une requête portant une version de contrat différente.
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    FEATURE_CONTRACT,
    POLITIQUE_CATEGORIES_INCONNUES,
    POLITIQUE_CATEGORIE_DEFAUT,
    RESOLUTION_LABELS,
)

# Version du format du document lui-même (pas du vocabulaire qu'il décrit).
FORMAT_CONTRAT = 1


# ═══════════════════════════════════════════════════════════════
#  CONSTRUCTION DEPUIS LES ARTEFACTS
# ═══════════════════════════════════════════════════════════════
def contrat_depuis_artefacts(ohe, scaler, train_columns, resolution_labels=None):
    """
    Dérive le contrat des artefacts entraînés. Aucune valeur n'est saisie
    manuellement : tout vient de `ohe.categories_`, `scaler.feature_names_in_`
    et `train_columns`.
    """
    labels = resolution_labels if resolution_labels is not None else RESOLUTION_LABELS

    categorielles = {}
    for nom, categories in zip(ohe.feature_names_in_, ohe.categories_):
        categorielles[str(nom)] = {
            "categories": [str(c) for c in categories],
            "unknown_policy": POLITIQUE_CATEGORIES_INCONNUES.get(
                str(nom), POLITIQUE_CATEGORIE_DEFAUT
            ),
        }

    contrat = {
        "format": FORMAT_CONTRAT,
        "generated_from": [
            "ohe_encoder.joblib",
            "scaler.joblib",
            "train_columns.joblib",
        ],
        "categorical_features": categorielles,
        "numeric_features": [str(c) for c in scaler.feature_names_in_],
        "train_columns_count": int(len(train_columns)),
        "resolution_labels": {str(k): str(v) for k, v in sorted(labels.items())},
    }
    contrat["contract_version"] = calculer_version(contrat)
    return contrat


def calculer_version(contrat):
    """
    Version = empreinte du contenu sémantique du contrat. Deux artefacts au
    vocabulaire identique donnent la même version ; le moindre ajout, retrait ou
    renommage de catégorie la change.
    """
    noyau = {
        cle: contrat[cle]
        for cle in (
            "format",
            "categorical_features",
            "numeric_features",
            "train_columns_count",
            "resolution_labels",
        )
    }
    brut = json.dumps(noyau, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(brut).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════
#  PERSISTANCE
# ═══════════════════════════════════════════════════════════════
def ecrire_contrat(contrat, chemin=FEATURE_CONTRACT):
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(contrat, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return chemin


def charger_contrat(chemin=FEATURE_CONTRACT):
    """Contrat versionné dans le dépôt, ou None s'il est absent."""
    if not os.path.isfile(chemin):
        return None
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
#  COMPARAISON
# ═══════════════════════════════════════════════════════════════
def comparer_contrats(attendu, obtenu):
    """
    Retourne la liste des divergences entre deux contrats (vide s'ils décrivent
    le même vocabulaire). Sert au démarrage de l'API et aux tests de dérive.
    """
    ecarts = []

    if attendu is None or obtenu is None:
        return ["contrat absent"]

    if attendu.get("contract_version") != obtenu.get("contract_version"):
        ecarts.append(
            f"contract_version : {attendu.get('contract_version')} != {obtenu.get('contract_version')}"
        )

    cat_a = attendu.get("categorical_features", {})
    cat_b = obtenu.get("categorical_features", {})
    for nom in sorted(set(cat_a) | set(cat_b)):
        if nom not in cat_a:
            ecarts.append(f"{nom} : feature catégorielle en trop")
            continue
        if nom not in cat_b:
            ecarts.append(f"{nom} : feature catégorielle manquante")
            continue
        va, vb = cat_a[nom].get("categories", []), cat_b[nom].get("categories", [])
        if va != vb:
            manquantes = sorted(set(va) - set(vb))
            en_trop = sorted(set(vb) - set(va))
            ecarts.append(f"{nom} : catégories manquantes={manquantes} en_trop={en_trop}")

    if attendu.get("numeric_features") != obtenu.get("numeric_features"):
        ecarts.append(
            f"numeric_features : {attendu.get('numeric_features')} != {obtenu.get('numeric_features')}"
        )

    if attendu.get("train_columns_count") != obtenu.get("train_columns_count"):
        ecarts.append(
            f"train_columns_count : {attendu.get('train_columns_count')} != {obtenu.get('train_columns_count')}"
        )

    return ecarts


# ═══════════════════════════════════════════════════════════════
#  INSPECTION D'UNE REQUÊTE
# ═══════════════════════════════════════════════════════════════
def inspecter_categories(contrat, valeurs):
    """
    Confronte les valeurs catégorielles d'une requête au vocabulaire du contrat.

    Retourne :
      {
        "unknown":   {feature: valeur_recue, ...},   # hors vocabulaire
        "alert":     [feature, ...],                 # divergence anormale
        "expected":  [feature, ...],                 # divergence documentée
        "known":     [feature, ...],
        "coverage":  0.0..1.0,                       # part de features reconnues
      }

    Note : `reason_x_policy` est dérivée par le feature engineering, elle n'est
    pas fournie par l'appelant — elle est ignorée ici et couverte par
    `Return_Reason` + `Within_Return_Policy`.
    """
    resultat = {"unknown": {}, "alert": [], "expected": [], "known": [], "coverage": 1.0}
    if not contrat:
        return resultat

    features = contrat.get("categorical_features", {})
    examinees = 0

    for nom, spec in features.items():
        if nom not in valeurs:
            continue
        examinees += 1
        valeur = valeurs[nom]
        if valeur in spec.get("categories", []):
            resultat["known"].append(nom)
            continue

        resultat["unknown"][nom] = valeur
        politique = spec.get("unknown_policy", POLITIQUE_CATEGORIE_DEFAUT)
        if politique == "expected":
            resultat["expected"].append(nom)
        else:
            resultat["alert"].append(nom)

    if examinees:
        resultat["coverage"] = round(len(resultat["known"]) / examinees, 4)

    return resultat
