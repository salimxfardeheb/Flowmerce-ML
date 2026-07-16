"""
Publie/actualise le dataset brut (data/raw) sur un repo Hugging Face de type dataset.

Usage :
    python scripts/push_dataset_to_hub.py --repo-id <username>/ecommerce_returns_smart_dataset
    python scripts/push_dataset_to_hub.py --repo-id <username>/... --private
    python scripts/push_dataset_to_hub.py --repo-id <username>/... --commit-message "Nouvelle version du dataset"

Le repo-id peut aussi être fourni via la variable d'environnement HF_DATASET_REPO.
Le token peut être fourni via HF_TOKEN, sinon celui mis en cache par
`hf auth login` est utilisé.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATASET, HF_DATASET_REPO, HF_TOKEN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=HF_DATASET_REPO,
        help="Identifiant du repo dataset Hugging Face (ex: username/ecommerce_returns_smart_dataset). "
             "Peut aussi être défini via HF_DATASET_REPO.",
    )
    parser.add_argument(
        "--token",
        default=HF_TOKEN,
        help="Token Hugging Face (Write access). Par défaut : token mis en cache par `hf auth login`.",
    )
    parser.add_argument(
        "--commit-message",
        default="Update raw dataset",
        help="Message de commit pour cette mise à jour.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Crée le repo en privé s'il n'existe pas encore.",
    )
    args = parser.parse_args()

    if not args.repo_id:
        parser.error("--repo-id est requis (ou définis HF_DATASET_REPO)")

    if not os.path.isfile(RAW_DATASET):
        parser.error(f"Fichier introuvable : {RAW_DATASET}")

    api = HfApi(token=args.token)

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    api.upload_file(
        path_or_fileobj=RAW_DATASET,
        path_in_repo=os.path.basename(RAW_DATASET),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
    )

    print(f"Dataset publié sur https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
