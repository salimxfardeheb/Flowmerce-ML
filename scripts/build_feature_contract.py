"""
Régénère `contracts/feature_contract.json` depuis les artefacts entraînés.

Appelé automatiquement à la fin de `src/pipeline.py`. À lancer à la main
uniquement si les artefacts ont été remplacés sans repasser par le pipeline
(téléchargement depuis Hugging Face, par exemple).

    python scripts/build_feature_contract.py
    python scripts/build_feature_contract.py --check   # ne réécrit rien, sort 1 si dérive

Après régénération, copier le fichier dans la web app :

    cp contracts/feature_contract.json ../flowmerce-web-app/lib/ml/feature-contract.json
"""

import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FEATURE_CONTRACT, MODEL_RESOLUTION, OHE_ENCODER, SCALER, TRAIN_COLUMNS
from src.feature_contract import (
    charger_contrat,
    comparer_contrats,
    contrat_depuis_artefacts,
    ecrire_contrat,
)


def construire():
    ohe = joblib.load(OHE_ENCODER)
    scaler = joblib.load(SCALER)
    train_columns = joblib.load(TRAIN_COLUMNS)
    return contrat_depuis_artefacts(ohe, scaler, train_columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Vérifie que le contrat versionné correspond aux artefacts, sans l'écrire.",
    )
    args = parser.parse_args()

    contrat = construire()

    if args.check:
        versionne = charger_contrat()
        ecarts = comparer_contrats(contrat, versionne)
        if ecarts:
            print("DÉRIVE — le contrat versionné ne correspond plus aux artefacts :")
            for e in ecarts:
                print(f"  - {e}")
            print("\nRelancer : python scripts/build_feature_contract.py")
            return 1
        print(f"Contrat à jour — version {contrat['contract_version']}")
        return 0

    ecrire_contrat(contrat)
    print(f"Contrat écrit → {FEATURE_CONTRACT}")
    print(f"  version              : {contrat['contract_version']}")
    print(f"  features catégorielles : {len(contrat['categorical_features'])}")
    print(f"  features numériques    : {len(contrat['numeric_features'])}")
    print(f"  colonnes d'entraînement: {contrat['train_columns_count']}")
    print("\nPenser à copier le fichier dans la web app :")
    print("  cp contracts/feature_contract.json ../flowmerce-web-app/lib/ml/feature-contract.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
