# Flowmerce — Système de Prédiction des Retours E-Commerce

Flowmerce est un système de machine learning qui prédit automatiquement, pour chaque demande de retour e-commerce :

1. **La résolution** — que faire du retour : Exchange, Refund, Reject, ou Repair
2. **Le payeur des frais de retour** — Client ou Vendeur

L'objectif est d'automatiser le traitement des retours pour réduire le temps de décision et standardiser les réponses.

---

## Architecture du projet

```
Flowmerce/
├── data/
│   ├── raw/
│   │   └── ecommerce_returns_smart_dataset.csv   # Dataset brut (50 000 lignes)
│   └── processed/
│       └── splits_encoded.pkl                    # Splits train/test encodés
│
├── models/
│   ├── model_resolution.joblib                   # Modèle Random Forest — Resolution
│   ├── model_shipping.joblib                     # Modèle Random Forest — Shipping_Paid_By
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
│   └── api.py                                    # API FastAPI v2.0.0 — endpoint /predict
│   
│
├── config.py                                     # Configuration centralisée (chemins, constantes)
├── requirements.txt
└── README.md
```

---

## Configuration centralisée (`config.py`)

Tous les chemins de fichiers et les constantes du projet sont définis dans `config.py` :

- Chemins vers les données, modèles et artefacts
- Colonnes à supprimer et colonnes catégorielles
- Seuils de performance (`SEUIL_F1_RESOLUTION`, `SEUIL_F1_SHIPPING`, `SEUIL_ACCURACY`)
- Mappings des labels (`RESOLUTION_MAP`, `SHIPPING_MAP`)
- Paramètres par défaut (split, SMOTE, random state, percentile de risque)

---

## Module de prétraitement partagé (`src/preprocessing.py`)

Source unique de vérité pour les transformations appliquées aussi bien en entraînement qu'en inférence.

- `appliquer_feature_engineering(df, seuil_risque)` — calcule les features engineerées
- `encoder_features(df, ohe, scaler, train_columns)` — encode avec les artefacts sauvegardés
- `preprocess(df, ohe, scaler, train_columns, seuil_risque)` — pipeline complet brut → prêt pour `.predict()`

Utilisé par `src/pipeline.py`, `api/api.py` et `test.py` pour garantir la cohérence train/inférence.

---

## Pipeline ML

### Étape 1 — Nettoyage (`src/pipeline.py`)

Colonnes supprimées car non prédictives ou identifiants :

| Colonne supprimée | Raison |
|---|---|
| `Order_ID`, `Customer_ID` | Identifiants, aucun signal |
| `Product_Name` | Trop granulaire |
| `Order_Date`, `Return_Date` | Remplacées par `Days_to_Return` |

> Note : `Customer_Satisfaction` et `Is_Suspicious` sont **gardés** comme features car ils apportent un signal discriminant pour les nouvelles features engineerées.

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
| `insatisfait_recurrent` | `Customer_Satisfaction <= 2 AND Customer_Past_Returns >= P75` | Client insatisfait récurrent → Repair/Exchange |

Le seuil P75 est calculé sur le dataset complet en phase d'entraînement, puis sauvegardé dans `models/training_params.joblib` pour être réutilisé à l'inférence.

### Étape 3 — Split train/test

- **80% train / 20% test** avec stratification combinée `Resolution × Shipping_Paid_By`
- Le split est fait **avant** l'encoding pour éviter le data leakage

### Étape 4 — Encoding

- **Targets** : mapping manuel vers entiers (Exchange=0, Refund=1, Reject=2, Repair=3)
- **Features catégorielles** : One-Hot Encoding fitté sur le train uniquement
- **Features numériques** : StandardScaler fitté sur le train uniquement
- Les artefacts `ohe_encoder.joblib`, `scaler.joblib` et `training_params.joblib` sont sauvegardés pour la production

### Étape 5 — Entraînement (Random Forest + RandomizedSearchCV)

Algorithme : **Random Forest** intégré dans un pipeline `imblearn` avec SMOTE appliqué uniquement sur les folds d'entraînement (pas sur le test).

Paramètres explorés (n_iter=30, cv=3, scoring=f1_weighted) :

```python
{
    "rf__n_estimators":      [100, 200, 300, 500],
    "rf__max_depth":         [None, 10, 20, 30],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf":  [1, 2, 4],
    "rf__max_features":      ["sqrt", "log2", 0.3],
    "rf__class_weight":      ["balanced", "balanced_subsample", None],
}
```

### Étape 6 — Validation des performances

Les modèles sont sauvegardés uniquement s'ils atteignent les seuils minimaux :

| Modèle | F1-score minimum | Accuracy minimum |
|---|---|---|
| Resolution | 0.70 | 0.70 |
| Shipping_Paid_By | 0.75 | 0.70 |

---

## Installation

```bash
git clone <repo>
cd Flowmerce
pip install -r requirements.txt
```

### Dépendances

```
pandas
numpy
scikit-learn
imbalanced-learn
joblib
fastapi
uvicorn
pydantic
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

### 2. Entraîner les modèles

```bash
python src/training.py
```

Génère (si les seuils de performance sont atteints) :
- `models/model_resolution.joblib`
- `models/model_shipping.joblib`
- `models/train_columns.joblib`

### 3. Tester les modèles directement

```bash
python test.py
```

Exécute des cas de test métier (fraude, retour normal, échange, réparation…) et affiche les prédictions avec probabilités, sans passer par l'API.

### 4. Lancer l'API

```bash
uvicorn api.api:app --reload --port 8000
```

L'interface Swagger est disponible sur `http://localhost:8000/docs`.

### 5. Tester l'API

```bash
python api/test_api.py
```

---

## API — Endpoints

### `GET /`
Retourne les endpoints disponibles.

### `GET /health`
Vérifie que les modèles et artefacts sont bien chargés.

```json
{
  "status": "ok",
  "models_loaded": {
    "resolution": true,
    "shipping": true
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

**Corps de la requête :**

```json
{
  "Customer_Gender": "Female",
  "Customer_Age": 30,
  "Customer_Wilaya": "Alger",
  "Customer_Past_Returns": 3,
  "Shop_Name": "TechnoStore",
  "Product_Category": "Electronics",
  "Product_Price_DA": 15000.0,
  "Order_Quantity": 1,
  "Total_Amount_DA": 15500.0,
  "Payment_Method": "CCP",
  "Shipping_Method": "Standard",
  "Shipping_Cost_DA": 500.0,
  "Return_Reason": "Defective",
  "Days_to_Return": 5,
  "Shop_Return_Window_Days": 14,
  "Within_Return_Policy": 1,
  "Fraud_Score": 12.0,
  "Customer_Satisfaction": 4,
  "Is_Suspicious": 0
}
```

**Réponse :**

```json
{
  "resolution": {
    "prediction": "Exchange",
    "probabilities": {
      "Exchange": 0.4821,
      "Refund": 0.2134,
      "Reject": 0.1872,
      "Repair": 0.1173
    }
  },
  "shipping_paid_by": {
    "prediction": "Vendeur",
    "probabilities": {
      "Client": 0.1243,
      "Vendeur": 0.8757
    }
  }
}
```

---

## Champs de la requête

| Champ | Type | Description |
|---|---|---|
| `Customer_Gender` | string | Genre du client |
| `Customer_Age` | int | Âge du client |
| `Customer_Wilaya` | string | Wilaya du client |
| `Customer_Past_Returns` | int | Nombre de retours passés |
| `Shop_Name` | string | Nom de la boutique |
| `Product_Category` | string | Catégorie du produit |
| `Product_Price_DA` | float | Prix du produit en DA |
| `Order_Quantity` | int | Quantité commandée |
| `Total_Amount_DA` | float | Montant total de la commande en DA |
| `Payment_Method` | string | Méthode de paiement |
| `Shipping_Method` | string | Transporteur |
| `Shipping_Cost_DA` | float | Frais de livraison en DA |
| `Return_Reason` | string | Raison du retour |
| `Days_to_Return` | int | Nombre de jours entre commande et retour |
| `Shop_Return_Window_Days` | int | Fenêtre de retour accordée par la boutique |
| `Within_Return_Policy` | int | Le retour est-il dans les délais ? (0 ou 1) |
| `Fraud_Score` | float | Score de fraude (0–100) |
| `Customer_Satisfaction` | int | Satisfaction client (1–5) |
| `Is_Suspicious` | int | Retour marqué comme suspect ? (0 ou 1) |
