import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    SPLITS_FILE,
    MODEL_RESOLUTION,
    TRAIN_COLUMNS,
    SEUIL_F1_RESOLUTION,
    SEUIL_ACCURACY,
    N_ITER_SEARCH,
    RANDOM_STATE,
    RESOLUTION_LABELS,
)


# ═══════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT (SMOTE + Random Forest)
# ═══════════════════════════════════════════════════════════════
def entrainer_random_forest(X_train, y_train, nom_modele="modele", n_iter=N_ITER_SEARCH):

    param_distributions = {
        "rf__n_estimators":      [100, 200, 300, 500],
        "rf__max_depth":         [None, 10, 20, 30],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf":  [1, 2, 4],
        "rf__max_features":      ["sqrt", "log2", 0.3],
        "rf__class_weight":      ["balanced", "balanced_subsample", None],
    }

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf",    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring="f1_weighted",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    dist = dict(pd.Series(y_train).value_counts())
    print(f"[Training] Distribution originale — {nom_modele} : {dist}")

    search.fit(X_train, y_train)

    print(f"[Training] Modèle '{nom_modele}' — meilleurs params :")
    for k, v in search.best_params_.items():
        print(f"           {k}: {v}")
    print(f"[Training] F1 CV (weighted) : {search.best_score_:.4f}\n")

    return search.best_estimator_


# ═══════════════════════════════════════════════════════════════
#  ÉVALUATION
# ═══════════════════════════════════════════════════════════════
def evaluer_modele(model, X_test, y_test, nom_modele="modele", labels=None):

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'=' * 50}")
    print(f"  ÉVALUATION — {nom_modele}")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"{'=' * 50}")

    print(f"\n[Classification Report]\n")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

    print(f"[Matrice de Confusion]")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=labels, columns=labels))
    print()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ═══════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCES
# ═══════════════════════════════════════════════════════════════
def afficher_feature_importances(model, feature_names, top_n=20):

    rf_model = model.named_steps["rf"]
    importances = pd.Series(
        rf_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print(f"\n{'─' * 50}")
    print(f"  TOP {top_n} FEATURE IMPORTANCES")
    print(f"{'─' * 50}")
    for i, (feat, imp) in enumerate(importances.head(top_n).items(), 1):
        bar = "█" * int(imp * 200)
        print(f"  {i:2d}. {feat:<35s} {imp:.4f}  {bar}")
    print(f"{'─' * 50}\n")

    return importances


# ═══════════════════════════════════════════════════════════════
#  DÉCISION DE PERFORMANCE
# ═══════════════════════════════════════════════════════════════
def verifier_performance(metrics, nom_modele, seuil_f1, seuil_acc=SEUIL_ACCURACY):

    f1  = metrics["f1"]
    acc = metrics["accuracy"]

    print(f"\n{'─' * 50}")
    print(f"  DÉCISION — {nom_modele}")
    print(f"{'─' * 50}")
    print(f"  F1-score  : {f1:.4f}  (seuil >= {seuil_f1})  {'PASS' if f1 >= seuil_f1 else 'FAIL'}")
    print(f"  Accuracy  : {acc:.4f}  (seuil >= {seuil_acc})  {'PASS' if acc >= seuil_acc else 'FAIL'}")

    if f1 >= seuil_f1 and acc >= seuil_acc:
        print(f"\n  >>> RÉSULTAT : Performance ACCEPTABLE — prêt pour la sauvegarde")
        print(f"{'─' * 50}\n")
        return True
    else:
        print(f"\n  >>> RÉSULTAT : Performance INSUFFISANTE — révision nécessaire")
        if f1 < seuil_f1:
            print(f"     - F1 trop bas")
        if acc < seuil_acc:
            print(f"     - Accuracy trop basse")
        print(f"{'─' * 50}\n")
        return False


def sauvegarder(objet, nom_fichier):
    joblib.dump(objet, nom_fichier)
    print(f"[Sauvegarde] {nom_fichier}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    t_total = time.time()

    print("=" * 60)
    print("  PIPELINE D'ENTRAÎNEMENT — FLOWMERCE (Random Forest)")
    print("=" * 60 + "\n")

    # --- Charger les splits encodés ---
    with open(SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)

    X_train     = splits["X_train"]
    X_test      = splits["X_test"]
    y_res_train = splits["y_res_train"]
    y_res_test  = splits["y_res_test"]

    # ── MODÈLE — Resolution ──
    print("\n" + "─" * 60)
    print("  MODÈLE : RESOLUTION (Random Forest)")
    print("─" * 60 + "\n")

    t1 = time.time()
    labels_res = list(RESOLUTION_LABELS.values())
    model_resolution = entrainer_random_forest(
        X_train, y_res_train, nom_modele="Resolution",
    )
    metrics_res = evaluer_modele(
        model_resolution, X_test, y_res_test,
        nom_modele="Resolution", labels=labels_res,
    )
    afficher_feature_importances(model_resolution, X_train.columns)
    res_ok = verifier_performance(
        metrics_res, "Resolution", seuil_f1=SEUIL_F1_RESOLUTION,
    )
    print(f"  Temps modèle Resolution : {time.time() - t1:.1f}s")

    # ── SAUVEGARDE CONDITIONNELLE ──
    print("\n" + "=" * 60)
    print("  BILAN FINAL")
    print("=" * 60)

    if res_ok:
        sauvegarder(model_resolution, MODEL_RESOLUTION)
        sauvegarder(list(X_train.columns), TRAIN_COLUMNS)
        print(f"\n  Modèle prêt pour le déploiement API !")
    else:
        print("[SKIP] model_resolution NON sauvegardé (performance insuffisante)")
        print(f"\n  Modèle n'a pas passé les seuils — révision nécessaire.")

    elapsed = time.time() - t_total
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\n  Temps total : {minutes}m {seconds:02d}s")
    print("=" * 60)