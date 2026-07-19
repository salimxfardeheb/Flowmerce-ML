"""
Publie/actualise le dataset réel sur un repo Hugging Face Hub (dataset).

Usage :
    python scripts/push_dataset_to_hub.py --repo-id <username>/flowmerce-real-dataset
    python scripts/push_dataset_to_hub.py --repo-id <username>/flowmerce-real-dataset --private
    python scripts/push_dataset_to_hub.py --repo-id <username>/flowmerce-real-dataset --commit-message "Ajout reclamations juillet"

Le repo-id peut aussi être fourni via la variable d'environnement HF_DATASET_REPO_ID
(ou dans un fichier .env à la racine du projet).
Le token peut être fourni via HF_TOKEN (idem), sinon celui mis en cache par
`huggingface-cli login` est utilisé.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATASET_REAL,
    HF_DATASET_REPO,
    HF_TOKEN,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=HF_DATASET_REPO,
        help="Identifiant du repo Hugging Face (ex: username/flowmerce-real-dataset). "
             "Peut aussi être défini via HF_DATASET_REPO_ID.",
    )
    parser.add_argument(
        "--token",
        default=HF_TOKEN,
        help="Token Hugging Face (Write access). Par défaut : token mis en cache par `huggingface-cli login`.",
    )
    parser.add_argument(
        "--commit-message",
        default="Update real dataset",
        help="Message de commit pour cette mise à jour.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Crée le repo en privé s'il n'existe pas encore.",
    )
    args = parser.parse_args()

    if not args.repo_id:
        parser.error("--repo-id est requis (ou définis HF_DATASET_REPO_ID)")

    if not os.path.isfile(RAW_DATASET_REAL):
        parser.error(f"Fichier introuvable : {RAW_DATASET_REAL}")

    api = HfApi(token=args.token)

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    api.upload_file(
        path_or_fileobj=RAW_DATASET_REAL,
        path_in_repo=os.path.basename(RAW_DATASET_REAL),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
    )

    print(f"Dataset publié sur https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
