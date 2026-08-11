"""
Compare le vocabulaire du dataset collecté à celui du modèle déployé.

Répond à une question opérationnelle : **est-ce que réentraîner apporterait
quelque chose ?** Tant que le dataset ne contient aucune valeur que le modèle
ignore, la réponse est non pour ce qui est du vocabulaire.

    python scripts/vocabulary_report.py
    python scripts/vocabulary_report.py --dataset data/raw/mon_export.csv
    python scripts/vocabulary_report.py --verite-terrain-seulement

Sortie : par feature catégorielle, les valeurs présentes dans le dataset et
absentes du contrat, avec leur nombre d'occurrences.

Code de sortie 0 dans tous les cas — c'est un rapport, pas une porte de CI.
"""

import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    COLONNE_LABEL_SOURCE,
    LABEL_SOURCES_VERITE_TERRAIN,
    RAW_DATASET_REAL,
)
from src.feature_contract import charger_contrat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=RAW_DATASET_REAL, help="CSV à analyser.")
    parser.add_argument(
        "--verite-terrain-seulement",
        action="store_true",
        help="N'analyser que les lignes exploitables à l'entraînement "
             f"({', '.join(LABEL_SOURCES_VERITE_TERRAIN)}).",
    )
    args = parser.parse_args()

    contrat = charger_contrat()
    if contrat is None:
        print("Contrat absent — lancer d'abord scripts/build_feature_contract.py")
        return 0

    if not os.path.isfile(args.dataset):
        print(f"Dataset introuvable : {args.dataset}")
        return 0

    df = pd.read_csv(args.dataset)
    total = len(df)

    if args.verite_terrain_seulement:
        if COLONNE_LABEL_SOURCE not in df.columns:
            print(f"Colonne {COLONNE_LABEL_SOURCE} absente — impossible de filtrer.")
            return 0
        origines = df[COLONNE_LABEL_SOURCE].astype(str).str.strip().str.lower()
        df = df[origines.isin(LABEL_SOURCES_VERITE_TERRAIN)]

    print("=" * 66)
    print("  VOCABULAIRE — dataset collecté  vs  modèle déployé")
    print("=" * 66)
    print(f"  dataset  : {args.dataset}")
    print(f"  lignes   : {len(df)} analysées sur {total}")
    print(f"  contrat  : {contrat['contract_version']}")

    if COLONNE_LABEL_SOURCE in df.columns:
        repartition = df[COLONNE_LABEL_SOURCE].value_counts().to_dict()
        exploitables = sum(
            n for src, n in repartition.items()
            if str(src).strip().lower() in LABEL_SOURCES_VERITE_TERRAIN
        )
        print(f"  labels   : {repartition}")
        print(f"  → exploitables à l'entraînement : {exploitables}")
    print()

    total_nouveau = 0

    for feature, spec in sorted(contrat["categorical_features"].items()):
        if feature not in df.columns:
            continue

        connues = set(spec["categories"])
        valeurs = Counter(df[feature].dropna().astype(str))
        nouvelles = {v: n for v, n in valeurs.items() if v not in connues}

        couverture = 1 - (sum(nouvelles.values()) / max(sum(valeurs.values()), 1))
        etat = "✓" if not nouvelles else ("·" if spec["unknown_policy"] == "expected" else "!")

        print(f"  {etat} {feature:<20} {len(connues):>3} apprises | "
              f"couverture {couverture:>6.1%}")

        if nouvelles:
            total_nouveau += len(nouvelles)
            for valeur, n in sorted(nouvelles.items(), key=lambda kv: -kv[1])[:12]:
                print(f"      + {valeur!r:<34} × {n}")
            if len(nouvelles) > 12:
                print(f"      … et {len(nouvelles) - 12} autres")

    print()
    print("-" * 66)
    if total_nouveau:
        print(f"  {total_nouveau} valeur(s) hors du vocabulaire appris.")
        print("  Un réentraînement les ferait entrer dans le modèle — à condition")
        print("  qu'elles portent des labels de vérité terrain en nombre suffisant.")
        print("  `·` = feature en cours de retrait, sa couverture n'a plus d'enjeu.")
    else:
        print("  Aucune valeur nouvelle : le vocabulaire du modèle couvre le dataset.")
    print("-" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
