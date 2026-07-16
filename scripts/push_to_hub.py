"""
Publie/actualise le contenu de `models/` sur un repo Hugging Face Hub.

Usage :
    python scripts/push_to_hub.py --repo-id <username>/flowmerce-resolution-model
    python scripts/push_to_hub.py --repo-id <username>/flowmerce-resolution-model --private
    python scripts/push_to_hub.py --repo-id <username>/flowmerce-resolution-model --commit-message "Retrain LightGBM v2"

Le repo-id peut aussi être fourni via la variable d'environnement HF_REPO_ID.
Le token peut être fourni via HF_TOKEN, sinon celui mis en cache par
`huggingface-cli login` est utilisé.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID"),
        help="Identifiant du repo Hugging Face (ex: username/flowmerce-resolution-model). "
             "Peut aussi être défini via HF_REPO_ID.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Token Hugging Face (Write access). Par défaut : token mis en cache par `huggingface-cli login`.",
    )
    parser.add_argument(
        "--commit-message",
        default="Update model artifacts",
        help="Message de commit pour cette mise à jour.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Crée le repo en privé s'il n'existe pas encore.",
    )
    args = parser.parse_args()

    if not args.repo_id:
        parser.error("--repo-id est requis (ou définis HF_REPO_ID)")

    if not os.path.isdir(MODELS_DIR):
        parser.error(f"Dossier introuvable : {MODELS_DIR}")

    api = HfApi(token=args.token)

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=MODELS_DIR,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )

    print(f"Modèles publiés sur https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
