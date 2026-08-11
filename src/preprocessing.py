"""
Module de prétraitement partagé — source unique de vérité.

Utilisé par :
  - api/api.py        (inférence temps réel)
  - test.py           (tests directs des modèles)
  - src/pipeline.py   (phase transform du feature engineering)

Ce module n'effectue AUCUN fit. Il applique les transformations
avec les artefacts déjà entraînés (OHE, Scaler).
"""

import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
def appliquer_feature_engineering(df, seuil_risque):
    """
    Applique exactement les mêmes features que le training.
    seuil_risque : P75 appris en entraînement, chargé depuis training_params.joblib.
    """
    df = df.copy()

    # --- Features de base ---
    df["ratio_delai_retour"] = (
        df["Days_to_Return"] / df["Shop_Return_Window_Days"]
    ).round(3)

    df["ratio_prix_livraison"] = (
        df["Product_Price_DA"] / (df["Shipping_Cost_DA"] + 1)
    ).round(2)

    df["client_a_risque"] = (
        df["Customer_Past_Returns"] >= seuil_risque
    ).astype(int)

    df["reason_x_policy"] = (
        df["Return_Reason"].astype(str) + "_" + df["Within_Return_Policy"].astype(str)
    )

    df["fraud_score_bin"] = pd.cut(
        df["Fraud_Score"],
        bins=[-1, 0, 30, 70, 100],
        labels=[0, 1, 2, 3],
    )
    df["fraud_score_bin"] = df["fraud_score_bin"].fillna(0).astype(int)

    # --- Features discriminantes ---
    df["fraud_x_suspicious"] = (
        df["Fraud_Score"] * df["Is_Suspicious"]
    ).round(2)

    df["hors_politique_fraud"] = (
        (df["Within_Return_Policy"] == 0) & (df["Fraud_Score"] > 50)
    ).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════
#  ENCODING (avec artefacts sauvegardés)
# ═══════════════════════════════════════════════════════════════
def encoder_features(df, ohe, scaler, train_columns):
    """
    Encode un DataFrame avec le OneHotEncoder et le StandardScaler
    sauvegardés à l'entraînement. Aligne les colonnes sur train_columns.

    Les colonnes à encoder sont lues sur l'encodeur lui-même
    (`ohe.feature_names_in_`), pas sur une liste de configuration. C'est le seul
    moyen d'être certain de lui présenter exactement les colonnes sur lesquelles
    il a été ajusté : une liste en dur et un artefact peuvent diverger, un
    artefact ne peut pas diverger de lui-même.

    Conséquence pratique : retirer une feature catégorielle du modèle ne demande
    qu'un réentraînement. L'inférence suit toute seule, sans changement de code
    ni fenêtre où les deux ne seraient pas d'accord.
    """
    cols_cat = ohe.feature_names_in_.tolist()
    cols_num = scaler.feature_names_in_.tolist()

    manquantes = [c for c in cols_cat if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes catégorielles absentes de l'entrée : {manquantes}. "
            f"L'encodeur chargé attend {cols_cat}."
        )

    # One-Hot Encoding
    ohe_feature_names = ohe.get_feature_names_out(cols_cat).tolist()
    df_ohe = pd.DataFrame(
        ohe.transform(df[cols_cat]),
        columns=ohe_feature_names,
        index=df.index,
    )

    # Standard Scaling
    df_num = pd.DataFrame(
        scaler.transform(df[cols_num]),
        columns=cols_num,
        index=df.index,
    )

    # Concaténation + alignement
    df_final = pd.concat([df_num, df_ohe], axis=1)
    df_final = df_final.reindex(columns=train_columns, fill_value=0)

    return df_final


# ═══════════════════════════════════════════════════════════════
#  PIPELINE COMPLET : brut → prêt pour prédiction
# ═══════════════════════════════════════════════════════════════
def preprocess(df, ohe, scaler, train_columns, seuil_risque):
    """
    Pipeline complet : feature engineering → encoding → alignement.
    Prend un DataFrame brut, retourne un DataFrame prêt pour .predict().

    seuil_risque est obligatoire — il doit venir de training_params.joblib.
    """
    df = appliquer_feature_engineering(df, seuil_risque=seuil_risque)
    X = encoder_features(df, ohe, scaler, train_columns)
    return X
