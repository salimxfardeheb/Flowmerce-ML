import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    RAW_DATASET,
    SPLITS_FILE,
    MODELS_DIR,
    PROCESSED_DIR,
    OHE_ENCODER,
    SCALER,
    TRAINING_PARAMS,
    COLONNES_A_SUPPRIMER,
    COLONNES_CATEGORIEL,
    RESOLUTION_MAP,
    SHIPPING_MAP,
    SEUIL_NA,
    TEST_SIZE,
    RANDOM_STATE,
    PERCENTILE_RISQUE,
)
from src.preprocessing import appliquer_feature_engineering


# ═══════════════════════════════════════════════════════════════
#  ÉTAPE 1 — NETTOYAGE DES DONNÉES
# ═══════════════════════════════════════════════════════════════
def nettoyer_donnees(df, colonnes_a_supprimer=None, seuil_na=SEUIL_NA):
    """
    Nettoyage :
    - Supprime les colonnes inutiles (identifiants, dates)
    - Supprime les doublons
    - Supprime les colonnes avec > seuil_na % de NA
    - Impute les NA restants (médiane / mode)
    """
    if colonnes_a_supprimer is None:
        colonnes_a_supprimer = []

    # 1) Supprimer colonnes inutiles
    colonnes_existantes = [c for c in colonnes_a_supprimer if c in df.columns]
    if colonnes_existantes:
        df = df.drop(columns=colonnes_existantes)
        print(f"[Nettoyage] Colonnes supprimées : {colonnes_existantes}")

    # 2) Suppression doublons
    avant = len(df)
    df = df.drop_duplicates()
    print(f"[Nettoyage] Doublons supprimés : {avant - len(df)} (de {avant} à {len(df)})")

    # 3) Suppression colonnes avec trop de NA (> seuil)
    total_na = df.isna().sum().sum()
    print(f"[Nettoyage] Valeurs manquantes totales : {total_na}")

    ratio_na = df.isna().mean()
    cols_trop_na = ratio_na[ratio_na > seuil_na].index.tolist()
    if cols_trop_na:
        print(f"[Nettoyage] Colonnes supprimées (>{seuil_na*100:.0f}% NA) : {cols_trop_na}")
        df = df.drop(columns=cols_trop_na)

    # 4) Imputation des NA restants
    cols_num = df.select_dtypes(include=[np.number]).columns
    cols_cat = df.select_dtypes(exclude=[np.number]).columns

    na_num = df[cols_num].isna().sum()
    na_num = na_num[na_num > 0]
    if len(na_num) > 0:
        for col in na_num.index:
            mediane = df[col].median()
            df[col] = df[col].fillna(mediane)
            print(f"[Nettoyage] Imputation médiane — {col} ({na_num[col]} NA → {mediane:.2f})")

    na_cat = df[cols_cat].isna().sum()
    na_cat = na_cat[na_cat > 0]
    if len(na_cat) > 0:
        for col in na_cat.index:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"[Nettoyage] Imputation mode — {col} ({na_cat[col]} NA → '{mode_val}')")

    print(f"[Nettoyage] Résultat : {df.shape[0]} lignes, {df.shape[1]} colonnes\n")
    return df


# ═══════════════════════════════════════════════════════════════
#  ÉTAPE 2 — FEATURE ENGINEERING
#
#  Le calcul du seuil P75 se fait ICI (phase fit).
#  La transformation utilise appliquer_feature_engineering() du
#  module partagé (preprocessing.py) pour garantir la cohérence.
# ═══════════════════════════════════════════════════════════════
def feature_engineering(df, percentile_risque=PERCENTILE_RISQUE):
    """
    Calcule le seuil P75 sur le dataset complet, puis applique
    le feature engineering via le module partagé.
    Retourne (df_transformé, seuil_risque).
    """
    # Calcul du seuil (phase fit — uniquement pendant le training)
    seuil = df["Customer_Past_Returns"].quantile(percentile_risque / 100)
    print(f"[FE] Seuil P{percentile_risque} calculé : {seuil:.0f}")

    # Transformation via le module partagé
    df = appliquer_feature_engineering(df, seuil_risque=seuil)

    # Logs
    print(f"[FE] ratio_delai_retour — min: {df['ratio_delai_retour'].min():.2f}, "
          f"max: {df['ratio_delai_retour'].max():.2f}, moy: {df['ratio_delai_retour'].mean():.2f}")
    print(f"[FE] client_a_risque — {df['client_a_risque'].sum()} clients à risque "
          f"({df['client_a_risque'].sum() / len(df) * 100:.1f}%)")
    print(f"[FE] reason_x_policy — {df['reason_x_policy'].nunique()} combinaisons uniques")
    print(f"[FE] fraud_score_bin — distribution: {df['fraud_score_bin'].value_counts().to_dict()}")
    print(f"[FE] fraud_x_suspicious — non-zéro: {(df['fraud_x_suspicious'] > 0).sum()} "
          f"({(df['fraud_x_suspicious'] > 0).sum() / len(df) * 100:.1f}%)")
    print(f"[FE] hors_politique_fraud — {df['hors_politique_fraud'].sum()} cas")
    print(f"[FE] insatisfait_recurrent — {df['insatisfait_recurrent'].sum()} cas\n")

    return df, seuil


# ═══════════════════════════════════════════════════════════════
#  ÉTAPE 3 — SPLIT TRAIN / TEST  (avant encoding)
# ═══════════════════════════════════════════════════════════════
def split_train_test(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):

    y_resolution = df["Resolution"]
    y_shipping   = df["Return_Shipping_Paid_By"]
    X = df.drop(columns=["Resolution", "Return_Shipping_Paid_By"])

    # Stratification combinée
    stratify_col = y_resolution.astype(str) + "_" + y_shipping.astype(str)

    X_train, X_test, y_res_train, y_res_test, y_ship_train, y_ship_test = train_test_split(
        X, y_resolution, y_shipping,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    print(f"[Split] X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"[Split] y_resolution  — train: {y_res_train.shape}, test: {y_res_test.shape}")
    print(f"[Split] y_shipping    — train: {y_ship_train.shape}, test: {y_ship_test.shape}\n")

    return X_train, X_test, y_res_train, y_res_test, y_ship_train, y_ship_test


# ═══════════════════════════════════════════════════════════════
#  ÉTAPE 4 — ENCODING  (fit sur train, transform sur test)
# ═══════════════════════════════════════════════════════════════
def encoder_targets(y_res_train, y_res_test, y_ship_train, y_ship_test):

    y_res_train  = y_res_train.map(RESOLUTION_MAP)
    y_res_test   = y_res_test.map(RESOLUTION_MAP)
    y_ship_train = y_ship_train.map(SHIPPING_MAP)
    y_ship_test  = y_ship_test.map(SHIPPING_MAP)

    print(f"[Encoding] Resolution       : {RESOLUTION_MAP}")
    print(f"[Encoding] Shipping_Paid_By : {SHIPPING_MAP}")

    return y_res_train, y_res_test, y_ship_train, y_ship_test


def encoder_features(X_train, X_test):
    """
    One-Hot Encoding avec OneHotEncoder (fit sur train uniquement).
    StandardScaler sur les colonnes numériques.
    Retourne X_train, X_test encodés + encoder + scaler pour la production.
    """
    cols_presentes = [c for c in COLONNES_CATEGORIEL if c in X_train.columns]
    cols_numeriques = [c for c in X_train.columns if c not in cols_presentes]

    # --- OneHotEncoder fitté sur train ---
    ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        drop=None,
    )
    ohe.fit(X_train[cols_presentes])

    ohe_feature_names = ohe.get_feature_names_out(cols_presentes).tolist()

    X_train_ohe = pd.DataFrame(
        ohe.transform(X_train[cols_presentes]),
        columns=ohe_feature_names,
        index=X_train.index,
    )
    X_test_ohe = pd.DataFrame(
        ohe.transform(X_test[cols_presentes]),
        columns=ohe_feature_names,
        index=X_test.index,
    )

    # --- StandardScaler sur les colonnes numériques ---
    scaler = StandardScaler()
    X_train_num = pd.DataFrame(
        scaler.fit_transform(X_train[cols_numeriques]),
        columns=cols_numeriques,
        index=X_train.index,
    )
    X_test_num = pd.DataFrame(
        scaler.transform(X_test[cols_numeriques]),
        columns=cols_numeriques,
        index=X_test.index,
    )

    # --- Concaténation ---
    X_train_final = pd.concat([X_train_num, X_train_ohe], axis=1)
    X_test_final  = pd.concat([X_test_num, X_test_ohe], axis=1)

    print(f"[Encoding] One-hot sur : {cols_presentes}")
    print(f"[Encoding] Scaler sur  : {len(cols_numeriques)} colonnes numériques")
    print(f"[Encoding] Colonnes finales — train: {X_train_final.shape[1]}, test: {X_test_final.shape[1]}\n")

    return X_train_final, X_test_final, ohe, scaler


# ═══════════════════════════════════════════════════════════════
#  MAIN — EXÉCUTION DU PIPELINE COMPLET
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 60)
    print("  PIPELINE DE PRÉTRAITEMENT — FLOWMERCE")
    print("=" * 60 + "\n")

    # Créer les dossiers si nécessaire
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Étape 1 : Nettoyage ---
    df = pd.read_csv(RAW_DATASET)
    df = nettoyer_donnees(df, colonnes_a_supprimer=COLONNES_A_SUPPRIMER, seuil_na=SEUIL_NA)

    # --- Étape 2 : Feature Engineering ---
    df, seuil_risque = feature_engineering(df, percentile_risque=PERCENTILE_RISQUE)

    # --- Étape 3 : Split (AVANT encoding) ---
    X_train, X_test, y_res_train, y_res_test, y_ship_train, y_ship_test = split_train_test(df)

    # --- Étape 4 : Encoding (fit sur train uniquement) ---
    y_res_train, y_res_test, y_ship_train, y_ship_test = encoder_targets(
        y_res_train, y_res_test, y_ship_train, y_ship_test
    )
    X_train, X_test, ohe, scaler = encoder_features(X_train, X_test)

    # --- Sauvegarde des splits ---
    splits = {
        "X_train": X_train,
        "X_test": X_test,
        "y_res_train": y_res_train,
        "y_res_test": y_res_test,
        "y_ship_train": y_ship_train,
        "y_ship_test": y_ship_test,
    }

    with open(SPLITS_FILE, "wb") as f:
        pickle.dump(splits, f)

    # --- Sauvegarde des artefacts de preprocessing ---
    joblib.dump(ohe, OHE_ENCODER)
    joblib.dump(scaler, SCALER)

    # Seuil P75 + métadonnées de training → artefact dédié
    training_params = {
        "seuil_risque": seuil_risque,
        "percentile_risque": PERCENTILE_RISQUE,
        "seuil_na": SEUIL_NA,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "colonnes_supprimees": COLONNES_A_SUPPRIMER,
    }
    joblib.dump(training_params, TRAINING_PARAMS)

    print("=" * 60)
    print("  PIPELINE TERMINÉ")
    print("=" * 60)
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Splits       → {SPLITS_FILE}")
    print(f"  OHE          → {OHE_ENCODER}")
    print(f"  Scaler       → {SCALER}")
    print(f"  Params       → {TRAINING_PARAMS}")
    print(f"  seuil_risque : {seuil_risque}")
    print("=" * 60)