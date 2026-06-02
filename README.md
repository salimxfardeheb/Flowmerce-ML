# Flowmerce — Système de Prédiction des Retours E-Commerce

Flowmerce est un système de machine learning qui prédit automatiquement, pour chaque demande de retour e-commerce, **la résolution à appliquer** : `Exchange`, `Refund`, `Reject` ou `Repair`.

L'objectif est d'automatiser le traitement des retours pour réduire le temps de décision et standardiser les réponses.

---

## Architecture du projet

```
Flowmerce/
├── data/
│   ├── raw/
│   │   └── ecommerce_returns_smart_dataset.csv   # Dataset brut
│   └── processed/
│       └── splits_encoded.pkl                    # Splits train/test encodés
│
├── models/
│   ├── model_resolution.joblib                   # Modèle LightGBM — Resolution
│   ├── train_columns.joblib                      # Liste des colonnes d'entraînement
│   ├── ohe_encoder.joblib                        # OneHotEncoder sauvegardé
│   ├── scaler.joblib                             # StandardScaler sauvegardé
│   └── training_params.joblib                    # Paramètres de training (seuil P75, etc.)
│
├── src/
│   ├── pipeline.py                               # Nettoyage, feature engineering, encoding
│   ├── preprocessing.py                          # Module partagé de prétraitement (inférence)
│   └── training.py                               # Entraînement, évaluation, sauvegarde
│
├── api/
│   ├── server.py                                 # API FastAPI v3.0.0 — endpoint /predict
│   └── .env                                       # Clé interne (INTERNAL_API_KEY)
│
├── logs/                                          # Rapports d'entraînement horodatés
│
├── config.py                                      # Configuration centralisée (chemins, constantes)
├── Dockerfile                                      # Image multi-stage (build + runtime non-root)
├── Dockerfile.prod                                # Image de production
├── docker-compose.yml                             # Services train + api
├── requirements.txt
└── README.md
```

---

## Configuration centralisée (`config.py`)

Tous les chemins de fichiers et les constantes du projet sont définis dans `config.py` :

- Chemins vers les données, modèles et artefacts
- Colonnes à supprimer (`COLONNES_A_SUPPRIMER`) et colonnes catégorielles (`COLONNES_CATEGORIEL`)
- Seuils de performance (`SEUIL_F1_RESOLUTION`, `SEUIL_ACCURACY`)
- Mapping des labels (`RESOLUTION_MAP`, `RESOLUTION_LABELS`)
- Paramètres par défaut (split, SMOTE, random state, percentile de risque, n_iter du grid search)

---

## Module de prétraitement partagé (`src/preprocessing.py`)

Source unique de vérité pour les transformations appliquées aussi bien en entraînement qu'en inférence. Ce module n'effectue **aucun fit** — il applique les artefacts déjà entraînés.

- `appliquer_feature_engineering(df, seuil_risque)` — calcule les features engineerées
- `encoder_features(df, ohe, scaler, train_columns)` — encode avec les artefacts sauvegardés et aligne les colonnes
- `preprocess(df, ohe, scaler, train_columns, seuil_risque)` — pipeline complet brut → prêt pour `.predict()`

Utilisé par `src/pipeline.py` (phase transform) et `api/server.py` (inférence) pour garantir la cohérence train/inférence.

---

## Pipeline ML

### Étape 1 — Nettoyage (`src/pipeline.py`)

Colonnes supprimées car non prédictives, identifiants, ou sources de data leakage :

| Colonne supprimée | Raison |
|---|---|
| `Order_ID`, `Customer_ID` | Identifiants, aucun signal |
| `Product_Name` | Trop granulaire |
| `Order_Date`, `Return_Date` | Remplacées par `Days_to_Return` |
| `Refund_Amount_DA` | Conséquence de la décision, pas une cause |
| `Customer_Satisfaction` | Data leakage (renseignée après résolution) |
| `Return_Shipping_Paid_By` | Ancienne cible, retirée du périmètre |

Le nettoyage inclut également :
- Suppression des doublons
- Suppression des colonnes avec > 30 % de valeurs manquantes
- Imputation des NA restants (médiane pour les numériques, mode pour les catégorielles)

### Étape 2 — Feature Engineering

| Feature créée | Formule | Intérêt |
|---|---|---|
| `ratio_delai_retour` | `Days_to_Return / Shop_Return_Window_Days` | Mesure si le retour est fait tôt ou tard dans la fenêtre |
| `ratio_prix_livraison` | `Product_Price_DA / (Shipping_Cost_DA + 1)` | Rapport valeur produit / coût logistique |
| `client_a_risque` | `Customer_Past_Returns >= P75` | Détecte les clients avec historique de retours élevé |
| `reason_x_policy` | `Return_Reason + "_" + Within_Return_Policy` | Interaction clé : même raison traitée différemment si hors délai |
| `fraud_score_bin` | Découpage en 4 bins : 0 / 1-30 / 31-70 / 71-100 | Discrétise le score de fraude |
| `fraud_x_suspicious` | `Fraud_Score × Is_Suspicious` | Amplification du signal Reject |
| `hors_politique_fraud` | `Within_Return_Policy == 0 AND Fraud_Score > 50` | Détecte les retours hors délai et suspects |

Le seuil P75 (`client_a_risque`) est calculé sur le dataset complet en phase d'entraînement, puis sauvegardé dans `models/training_params.joblib` pour être réutilisé à l'inférence.

### Étape 3 — Split train/test

- **80 % train / 20 % test** avec stratification sur `Resolution`
- Le split est fait **avant** l'encoding pour éviter le data leakage

### Étape 4 — Encoding

- **Target** : mapping manuel vers entiers (`Exchange=0`, `Refund=1`, `Reject=2`, `Repair=3`)
- **Features catégorielles** : One-Hot Encoding fitté sur le train uniquement (`handle_unknown="ignore"`)
- **Features numériques** : StandardScaler fitté sur le train uniquement
- Les artefacts `ohe_encoder.joblib`, `scaler.joblib` et `training_params.joblib` sont sauvegardés pour la production

### Étape 5 — Entraînement (`src/training.py`)

Algorithme : **LightGBM** (`LGBMClassifier`, objectif `multiclass`, `class_weight="balanced"`) intégré dans un pipeline `imblearn` avec **SMOTE** appliqué uniquement sur les folds d'entraînement.

L'entraînement se fait en **deux phases** pour rester rapide :

- **Phase 1 — Grid search** : `RandomizedSearchCV` (`n_iter=30`, `cv=5`, scoring `f1_weighted`) sur un sous-échantillon de **30 000 lignes** (sans refit).
- **Phase 2 — Refit final** : ré-entraînement du meilleur jeu d'hyperparamètres sur **100 %** des données d'entraînement.

Paramètres explorés :

```python
{
    "lgbm__n_estimators":      [200, 400, 600],
    "lgbm__num_leaves":        [31, 63, 127],
    "lgbm__max_depth":         [-1, 12, 20],
    "lgbm__learning_rate":     [0.02, 0.05, 0.1],
    "lgbm__min_child_samples": [10, 20, 50],
    "lgbm__subsample":         [0.8, 1.0],
    "lgbm__colsample_bytree":  [0.8, 1.0],
}
```

À la fin de l'entraînement, un **rapport horodaté** (`logs/training_AAAAMMJJ_HHMMSS.txt`) est généré avec les métriques, le classification report, la matrice de confusion et les temps d'exécution par phase.

### Étape 6 — Validation des performances

Le modèle n'est sauvegardé que s'il atteint les seuils minimaux :

| Modèle | F1-score minimum | Accuracy minimum |
|---|---|---|
| Resolution | 0.62 | 0.65 |

---

## Installation

```bash
git clone <repo>
cd Flowmerce-ML
pip install -r requirements.txt
```

### Dépendances

```
pandas
numpy
scikit-learn
lightgbm
imbalanced-learn
joblib
fastapi
uvicorn
pydantic
python-dotenv
```

---

## Utilisation

### 1. Lancer le pipeline de prétraitement

```bash
python src/pipeline.py
```

Génère :
- `data/processed/splits_encoded.pkl`
- `models/ohe_encoder.joblib`
- `models/scaler.joblib`
- `models/training_params.joblib` (seuil P75 et métadonnées)

### 2. Entraîner le modèle

```bash
python src/training.py
```

Génère (si les seuils de performance sont atteints) :
- `models/model_resolution.joblib`
- `models/train_columns.joblib`
- `logs/training_<horodatage>.txt` (rapport d'entraînement)

### 3. Lancer l'API

Créez d'abord le fichier `api/.env` à partir de l'exemple :

```bash
cp .env.example api/.env   # contient INTERNAL_API_KEY=...
```

Puis lancez le serveur :

```bash
uvicorn api.server:app --reload --port 8000
```

L'interface Swagger est disponible sur `http://localhost:8000/docs`.

---

## Docker

Le projet fournit une image **multi-stage** (builder + runtime non-root, avec healthcheck) et un `docker-compose.yml` avec deux services.

```bash
# Entraînement (pipeline + training) dans un container
docker compose run --rm train

# Lancer l'API (port 8000 par défaut, configurable via API_PORT)
docker compose up api
```

Variables d'environnement utiles :
- `INTERNAL_API_KEY` (dans `api/.env`) — clé d'authentification de l'API
- `API_PORT` — port exposé pour l'API (défaut `8000`)
- `ENVIRONMENT` — `development` / `production`

---

## API — Endpoints

L'API est en version **3.0.0**. L'endpoint `/predict` est protégé par une clé interne passée dans l'en-tête HTTP **`X-Internal-Key`**.

### `GET /`
Retourne les endpoints disponibles et la version.

### `GET /health`
Vérifie que le modèle et les artefacts sont bien chargés.

```json
{
  "status": "ok",
  "models_loaded": {
    "resolution": true
  },
  "artifacts_loaded": {
    "ohe_encoder": true,
    "scaler": true,
    "train_columns": true,
    "training_params": true
  },
  "seuil_risque": 5.0
}
```

### `POST /predict`

> En-tête requis : `X-Internal-Key: <votre_clé>`

**Corps de la requête :**

```json
{
  "Customer_Gender": "Female",
  "Customer_Age": 34,
  "Customer_Wilaya": "Alger",
  "Customer_Past_Returns": 1,
  "Shop_Name": "Shop_001",
  "Product_Category": "Vetements",
  "Product_Price_DA": 3500.0,
  "Order_Quantity": 1,
  "Total_Amount_DA": 3500.0,
  "Payment_Method": "Especes livraison",
  "Shipping_Method": "Yalidine",
  "Shipping_Cost_DA": 400.0,
  "Return_Reason": "Mauvaise taille",
  "Days_to_Return": 4,
  "Shop_Return_Window_Days": 14,
  "Within_Return_Policy": 1,
  "Fraud_Score": 5.0
}
```

> `Is_Suspicious` n'est **pas** envoyé par le client : il est calculé automatiquement côté serveur (`Fraud_Score >= 60`).
> `Customer_Satisfaction` n'est plus accepté (retiré pour éviter le data leakage).

**Réponse :**

```json
{
  "resolution": {
    "prediction": "Exchange",
    "confidence": 0.4821,
    "probabilities": {
      "Exchange": 0.4821,
      "Refund": 0.2134,
      "Reject": 0.1872,
      "Repair": 0.1173
    }
  },
  "risk_flag": {
    "is_suspicious": false,
    "fraud_score": 5.0,
    "seuil_risque": 5.0,
    "above_threshold": true
  }
}
```

---

## Champs de la requête

| Champ | Type | Contrainte | Description |
|---|---|---|---|
| `Customer_Gender` | string | — | Genre du client |
| `Customer_Age` | int | — | Âge du client |
| `Customer_Wilaya` | string | — | Wilaya du client |
| `Customer_Past_Returns` | int | `>= 0` | Nombre de retours passés |
| `Shop_Name` | string | — | Nom de la boutique |
| `Product_Category` | string | — | Catégorie du produit |
| `Product_Price_DA` | float | `> 0` | Prix du produit en DA |
| `Order_Quantity` | int | `>= 1` | Quantité commandée |
| `Total_Amount_DA` | float | `> 0` | Montant total de la commande en DA |
| `Payment_Method` | string | — | Méthode de paiement |
| `Shipping_Method` | string | — | Transporteur |
| `Shipping_Cost_DA` | float | `>= 0` | Frais de livraison en DA |
| `Return_Reason` | string | — | Raison du retour |
| `Days_to_Return` | int | `>= 0` | Nombre de jours entre commande et retour |
| `Shop_Return_Window_Days` | int | `> 0` | Fenêtre de retour accordée par la boutique |
| `Within_Return_Policy` | int | `0` ou `1` | Le retour est-il dans les délais ? |
| `Fraud_Score` | float | `0`–`100` | Score de fraude |
