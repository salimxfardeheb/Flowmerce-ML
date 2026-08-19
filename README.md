# Flowmerce — Système de Prédiction des Retours E-Commerce

Flowmerce est un système de machine learning qui prédit automatiquement, pour chaque demande de retour e-commerce, **la résolution à appliquer** : `Exchange`, `Reject` ou `Repair`.

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
│   ├── feature_contract.py                       # Construction / lecture du contrat de features
│   └── training.py                               # Entraînement, évaluation, sauvegarde
│
├── contracts/
│   └── feature_contract.json                     # Contrat de features versionné (vocabulaire servi)
│
├── api/
│   ├── server.py                                 # API FastAPI v5.0.0 — /predict, /save_claim, /feature-contract
│   └── .env                                       # Clé interne (INTERNAL_API_KEY)
│
├── scripts/
│   ├── build_feature_contract.py                 # Régénère / vérifie (--check) le contrat de features
│   ├── push_models.py                            # Publie les artefacts sur Hugging Face Hub
│   ├── push_dataset.py                           # Publie le dataset collecté sur Hugging Face Hub
│   └── vocabulary_report.py                      # Compare le vocabulaire collecté à celui du modèle
│
├── tests/
│   ├── conftest.py                               # Fixtures (artefacts neutralisés, CSV temporaire)
│   ├── test_auth.py                              # Authentification fail-closed (401 / 403 / 503)
│   ├── test_feature_contract.py                  # Le contrat versionné correspond aux artefacts
│   ├── test_ground_truth.py                      # Collecte de la vérité terrain (/save_claim)
│   ├── test_predict_contract.py                  # Contrat de /predict (schéma, 409 de version)
│   ├── test_save_claim.py                        # Contrat de /save_claim
│   └── test_train_serve_skew.py                  # Cohérence entraînement / inférence
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
| `Refund_Amount_DA` | Conséquence de la décision, pas une cause — **plus collectée** |
| `Customer_Satisfaction` | Data leakage (renseignée après résolution) |
| `Return_Shipping_Paid_By` | Ancienne cible, retirée du périmètre — **plus collectée** |

> `Refund_Amount_DA` et `Return_Shipping_Paid_By` ne sont plus ni acceptés par
> `/save_claim`, ni écrits dans le dataset. Ils restent listés dans
> `COLONNES_A_SUPPRIMER` uniquement parce que les datasets historiques les
> contiennent encore : le drop est conditionnel et garantit qu'ils ne
> repassent jamais en feature. Ne pas les retirer de cette liste tant qu'un
> dataset les porte.

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

- **Target** : mapping manuel vers entiers (`Exchange=0`, `Reject=1`, `Repair=2`)
- **Features catégorielles** : One-Hot Encoding fitté sur le train uniquement (`handle_unknown="ignore"`)
- **Features numériques** : StandardScaler fitté sur le train uniquement
- Les artefacts `ohe_encoder.joblib`, `scaler.joblib` et `training_params.joblib` sont sauvegardés pour la production

### Étape 5 — Entraînement (`src/training.py`)

Algorithme : **LightGBM** (`LGBMClassifier`, objectif `multiclass`, `class_weight="balanced"`) intégré dans un pipeline `imblearn` avec **SMOTE** appliqué uniquement sur les folds d'entraînement.

L'entraînement se fait en **deux phases** pour rester rapide :

- **Phase 1 — Grid search** : `RandomizedSearchCV` (`n_iter=5`, `cv=5`, scoring `f1_weighted`) sur un sous-échantillon de **12 000 lignes** (sans refit).
- **Phase 2 — Refit final** : ré-entraînement du meilleur jeu d'hyperparamètres sur **100 %** des données d'entraînement.

`n_iter` est piloté par `N_ITER_SEARCH` (`config.py`) et la taille du sous-échantillon par `TUNE_SAMPLE` (`src/training.py`) — augmente-les pour une recherche plus large.

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
| Resolution | 0.68 | 0.70 |


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
python src/pipeline.py                              # dataset portant la colonne Label_Source
python src/pipeline.py --origine-labels synthetic   # dataset simulé, sans colonne de provenance
```

Génère :

- `data/processed/splits_encoded.pkl`
- `models/ohe_encoder.joblib`
- `models/scaler.joblib`
- `models/training_params.joblib` (seuil P75 et métadonnées)

**Filtrage de la vérité terrain.** Avant tout nettoyage, le pipeline ne conserve que les lignes dont
`Label_Source` vaut `human`, `policy_rule` ou `synthetic`. Les lignes `model` — des résolutions
écrites par le modèle lui-même — sont **exclues** : sans ce filtre, le modèle se réentraîne sur ses
propres sorties et sédimente ses biais. Un dataset sans colonne de provenance est refusé, sauf
origine déclarée explicitement via `--origine-labels`. Un dataset ne contenant que des labels `model`
interrompt l'entraînement avec un message actionnable, plutôt que de produire un modèle sur du vide.

### 2. Entraîner le modèle

```bash
python src/training.py
```

Génère (si les seuils de performance sont atteints) :

- `models/model_resolution.joblib`
- `models/train_columns.joblib`
- `contracts/feature_contract.json` (contrat de features — voir ci-dessous)
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

### 4. Publier les artefacts sur Hugging Face Hub

```bash
huggingface-cli login   # une seule fois, token avec accès Write
python scripts/push_models.py --repo-id <username>/flowmerce-resolution-model
```

Le repo-id peut aussi être défini via la variable d'environnement `HF_REPO_ID`.

### 5. Publier le dataset collecté sur Hugging Face Hub

Les réclamations réelles écrites par `/save_claim` dans `data/raw/` sont poussées
vers le dépôt dataset (`HF_DATASET_REPO` / `HF_DATASET_FILE`) :

```bash
python scripts/push_dataset.py
```

C'est ce dataset que le pipeline relit à l'entraînement suivant : il ferme la
boucle collecte → réentraînement décrite plus bas.

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

## Contrat de features (`contracts/feature_contract.json`)

Le vocabulaire catégoriel du modèle est **dérivé des artefacts entraînés**, jamais saisi à la main :

```text
ohe_encoder.joblib + scaler.joblib + train_columns.joblib
                    │
                    ▼   contrat_depuis_artefacts()
        contracts/feature_contract.json     (version = empreinte du contenu)
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      API ML          web app (lib/ml/feature-contract.json, copie identique)
```

Le fichier est régénéré par `src/training.py` à chaque modèle retenu, ou à la main :

```bash
python scripts/build_feature_contract.py           # régénère
python scripts/build_feature_contract.py --check   # sort 1 si le contrat versionné a dérivé
cp contracts/feature_contract.json ../flowmerce-web-app/lib/ml/feature-contract.json
```

Quatre garde-fous rendent toute divergence bruyante :

| Garde-fou | Effet |
|---|---|
| `tests/test_feature_contract.py` | échoue si le contrat versionné ne correspond plus aux artefacts |
| `tests/test_train_serve_skew.py` | échoue si la copie web app diverge du contrat ML |
| Démarrage de l'API | **refuse de servir** si le contrat ne décrit pas les artefacts chargés |
| En-tête `X-Feature-Contract-Version` | **409** si l'appelant est construit sur un autre vocabulaire |

À l'inférence, une valeur hors vocabulaire n'est plus convertie en vecteur nul silencieux : elle est
journalisée (`contract.categories_inconnues`) et remontée dans la réponse sous `contract`. La politique
par feature est portée par le contrat : `expected` pour les features en cours de retrait
(signalées, jamais alertées), `alert` pour toutes les autres.

`models/` est dans le `.gitignore` de ce dépôt (c'est H-07, le build ML non reproductible), mais
`contracts/feature_contract.json` **est versionné**. Le vocabulaire servi devient donc lisible
dans l'historique Git alors que les artefacts binaires ne le sont pas — un changement de
vocabulaire apparaîtra dans une revue de code.

### Features en cours de retrait

`Shop_Name` et `Shipping_Method` ne seront plus des features au prochain réentraînement
(`COLONNES_A_SUPPRIMER`). Elles restent **collectées** par `/save_claim` — le dataset doit
décrire la réalité d'une réclamation — mais leur vocabulaire n'a plus d'enjeu, d'où
`unknown_policy: "expected"`.

| Feature | Pourquoi elle sort |
|---|---|
| `Shop_Name` | Cardinalité non bornée, qui croît avec le nombre de vendeurs. Chaque nouvelle boutique était hors vocabulaire à vie : le one-hot ne peut pas apprendre une liste ouverte. Le signal utile d'une boutique (taux de retour historique, ancienneté, volume) se porte par des features numériques, qui ne périment pas. |
| `Shipping_Method` | Vocabulaire vivant — les transporteurs algériens apparaissent et disparaissent — pour un apport marginal : le rapport d'entraînement plaçait `Shipping_Method_Yalidine` au 19ᵉ rang des importances. |

Le retrait ne demande **aucun changement de code à l'inférence** :
`src/preprocessing.encoder_features` lit les colonnes à encoder sur `ohe.feature_names_in_`, donc
sur le modèle réellement chargé. Le modèle actuel les connaît encore et continue de servir ; le
prochain ne les connaîtra plus et servira tout autant.

### Vocabulaire nouveau — `/save_claim` collecte, `/predict` interroge

Les deux endpoints traitent une valeur hors vocabulaire de façon **volontairement opposée** :

| | `/predict` | `/save_claim` |
|---|---|---|
| Nature | interroge un modèle figé | décrit une réclamation réelle |
| Valeur inconnue | anomalie → `WARNING`, remontée dans `contract` | **matière première** → `INFO`, conservée telle quelle |
| Effet | vecteur d'entrée partiel, signalé | aucune perte : la valeur entre au dataset |

C'est ce qui permet au vocabulaire de s'enrichir : une wilaya, une catégorie ou un moyen de
paiement que le modèle ignore est **collecté sans déformation**, et devient apprenable au
prochain entraînement. Pour savoir si ce moment est venu :

```bash
python scripts/vocabulary_report.py                           # dataset collecté vs modèle
python scripts/vocabulary_report.py --verite-terrain-seulement
```

Le rapport liste, par feature, les valeurs présentes dans le dataset et absentes du contrat avec
leur nombre d'occurrences, et rappelle combien de lignes portent un label exploitable. Une valeur
nouvelle n'a d'intérêt à l'entraînement que si elle est accompagnée de décisions humaines.

### Cycle complet d'enrichissement du vocabulaire

```text
nouvelles valeurs collectées (/save_claim)
        ↓  scripts/vocabulary_report.py  →  « 12 wilayas inconnues, 340 labels humains »
décisions humaines suffisantes ?
        ↓  python src/pipeline.py        →  encodeur ajusté sur le nouveau vocabulaire
        ↓  python src/training.py        →  modèle + contrat régénérés (nouvelle version)
        ↓  cp contracts/feature_contract.json ../flowmerce-web-app/lib/ml/feature-contract.json
        ↓  déployer les deux ensemble
```

L'ordre importe : déployer l'API ML sans répercuter la copie web app fait répondre **409** à
chaque prédiction — un arrêt franc, pas une dérive silencieuse.

---

## API — Endpoints

L'API est en version **5.0.0**. Les endpoints `/predict`, `/save_claim` et `/feature-contract` sont protégés par une clé interne passée dans l'en-tête HTTP **`X-Internal-Key`** ; `/` et `/health` sont publics.

### Authentification — fail-closed

| Situation | Réponse |
|---|---|
| `INTERNAL_API_KEY` non configuré | **503** sur toute requête authentifiée, et le processus **refuse de démarrer** |
| En-tête absent ou vide | **401** |
| En-tête erroné | **403** (comparaison à temps constant, `secrets.compare_digest`) |
| En-tête correct | requête servie |

Une variable d'environnement absente ne peut donc plus ouvrir l'API : `INTERNAL_KEY` valait `None`,
l'en-tête absent aussi, et la comparaison `None != None` étant fausse, la garde laissait passer.

> Vue d'ensemble de la sécurité des trois dépôts : [`../SECURITE.md`](../SECURITE.md).

### `GET /`

Retourne les endpoints disponibles et la version.

### `GET /health`

Vérifie que le modèle et les artefacts sont bien chargés.

```json
{
  "status": "ok",
  "source_artefacts": "huggingface",
  "models_loaded": {
    "resolution": true
  },
  "artifacts_loaded": {
    "ohe_encoder": true,
    "scaler": true,
    "train_columns": true,
    "training_params": true
  },
  "seuil_risque": 3.0
}
```

> `source_artefacts` vaut `huggingface` ou `local` selon `USE_HF_MODELS` (`config.py`).

### `GET /feature-contract`

> En-tête requis : `X-Internal-Key: <votre_clé>`

Retourne le contrat de features servi par cette instance : vocabulaire catégoriel
appris, features numériques et `contract_version`. C'est le document de référence
que la web app copie dans `lib/ml/feature-contract.json` ; comparer les deux
suffit à détecter une divergence avant qu'un `/predict` ne parte en **409**.

Répond **503** si le contrat n'a pas pu être chargé au démarrage.

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
  "Fraud_Score": 5.0,
  "Is_Suspicious": 0
}
```

> **En-tête recommandé :** `X-Feature-Contract-Version: <version>` — l'API répond **409** si l'appelant
> a été construit sur un autre vocabulaire de features. Mieux vaut un refus explicite qu'une prédiction
> rendue sur des colonnes mal alignées.
>
> `Is_Suspicious` est **fourni par l'appelant** et n'est plus recalculé côté serveur. Il portait
> auparavant trois définitions incompatibles : `Customer_Past_Returns >= fraudReturnThreshold`
> côté web app (persistée dans le dataset), `Fraud_Score >= 60` à l'inférence, et rien du tout dans
> le schéma Pydantic — qui l'écartait en silence. C'est désormais le seuil configuré par le vendeur
> qui fait foi, de bout en bout.
>
> `Customer_ID` est accepté mais n'est pas une feature (il figure dans `COLONNES_A_SUPPRIMER`).
> Tout autre champ provoque un **422** : `ReturnRequest` est en `extra="forbid"`, une divergence de
> schéma est bruyante au lieu de faire disparaître une feature sans trace.
>
> `Customer_Satisfaction` n'est plus accepté (retiré pour éviter le data leakage).

**Réponse :**

```json
{
  "resolution": {
    "prediction": "Exchange",
    "confidence": 0.9435,
    "probabilities": {
      "Exchange": 0.9435,
      "Reject": 0.0491,
      "Repair": 0.0074
    }
  },
  "risk_flag": {
    "is_suspicious": false,
    "fraud_score": 5.0,
    "seuil_risque": 3.0,
    "client_a_risque": false
  },
  "contract": {
    "version": "d720ad897bf56f11",
    "degraded": true,
    "unknown_categories": { "Shop_Name": "ia-store" },
    "alert_features": [],
    "expected_unknown": ["Shop_Name"],
    "categorical_coverage": 0.8571
  }
}
```

| Champ de la réponse | Signification |
|---|---|
| `resolution.prediction` | Résolution prédite : `Exchange`, `Reject` ou `Repair` |
| `resolution.confidence` | Probabilité de la classe retenue (= max des `probabilities`) |
| `risk_flag.is_suspicious` | Valeur d'`Is_Suspicious` transmise par l'appelant (seuil vendeur) |
| `risk_flag.seuil_risque` | Seuil P75 appris à l'entraînement (`training_params.joblib`) |
| `risk_flag.client_a_risque` | `Customer_Past_Returns >= seuil_risque` |
| `contract.degraded` | Au moins une valeur catégorielle était hors vocabulaire : le vecteur d'entrée est partiel |
| `contract.unknown_categories` | Les valeurs concernées, feature par feature |
| `contract.alert_features` | Sous-ensemble anormal (hors divergences documentées comme `Shop_Name`) |
| `contract.categorical_coverage` | Part des groupes catégoriels reconnus, 0..1 |

### `POST /save_claim`

> En-tête requis : `X-Internal-Key: <votre_clé>`

Insère une réclamation **réelle**, résolution finale comprise, dans le dataset
d'entraînement (`data/raw/ecommerce_returns_real_dataset.csv`). Réponse `201`.

**Idempotence.** Un `Order_ID` déjà présent renvoie `200` avec `status: "duplicate"` sans rien
insérer : un export rejoué ou un retry réseau ne duplique plus la ligne. Les écritures concurrentes
sont sérialisées par un verrou exclusif sur le fichier.

**`Label_Source` est obligatoire** — `human`, `policy_rule` ou `model`. Une résolution sans
provenance est indiscernable d'une sortie de modèle ; les lignes `model` sont conservées pour la
traçabilité mais exclues de l'entraînement. Un dataset existant qui ne porte pas la colonne fait
répondre `409` : `extrasaction="ignore"` ferait sinon disparaître la provenance en silence.

**Corps de la requête :**

```json
{
  "Order_ID": "cmrxl0wm9000023zzv2lryhrn",
  "Customer_ID": "CUST-501",
  "Customer_Age": 30,
  "Customer_Gender": "Unknown",
  "Customer_Wilaya": "Alger",
  "Customer_Past_Returns": 0,
  "Shop_Name": "ia-store",
  "Product_Category": "Vetements",
  "Product_Name": "OVERSIZE VINTAGE SHIRT",
  "Product_Price_DA": 5500.0,
  "Order_Quantity": 1,
  "Total_Amount_DA": 5500.0,
  "Payment_Method": "Especes livraison",
  "Shipping_Method": "Yalidine",
  "Shipping_Cost_DA": 400.0,
  "Order_Date": "2026-07-23",
  "Return_Date": "2026-07-31",
  "Days_to_Return": 8,
  "Shop_Return_Window_Days": 14,
  "Within_Return_Policy": 1,
  "Return_Reason": "Mauvaise taille",
  "Resolution": "Exchange",
  "Label_Source": "human",
  "Fraud_Score": 5.0,
  "Is_Suspicious": 0,
  "Customer_Satisfaction": 3
}
```

> `Resolution` accepte `Exchange`, `Reject`, `Repair` ou `Refund`.
> `Customer_Satisfaction` est optionnel (`1`–`5`).

**Champs retirés du contrat :**

`Return_Shipping_Paid_By` et `Refund_Amount_DA` **ne sont plus acceptés** : aucun
point d'entrée du produit ne les collectait, la valeur envoyée était constante
(`""` et `0`) et n'apportait aucun signal. Ils ne sont plus écrits dans le
dataset.

Pendant la fenêtre de transition, `ReclamationInput` est configuré en
`extra="ignore"` : un ancien client qui envoie encore ces clés reçoit un `201`
et les valeurs sont simplement ignorées, plutôt qu'un `422`. **À repasser en
`extra="forbid"` une fois tous les clients migrés.**

**Réponse :**

```json
{
  "status": "ok",
  "message": "Réclamation insérée avec succès.",
  "order_id": "cmrxl0wm9000023zzv2lryhrn"
}
```

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

Les tests neutralisent le chargement des artefacts Hugging Face à l'import :
aucun réseau ni token n'est nécessaire.

---

## Champs de la requête

Endpoint `/predict` :

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
