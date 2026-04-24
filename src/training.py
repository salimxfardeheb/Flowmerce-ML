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

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION MACHINE
#  i5-1135G7 : 4 cœurs physiques / 8 threads logiques
# ─────────────────────────────────────────────────────────────
N_JOBS_RF     = -1   # RF utilise tous les cœurs pour construire les arbres
N_JOBS_SEARCH = 2    # 2 folds en parallèle max — évite la surcharge mémoire
TUNE_SAMPLE   = 15000  # lignes pour le grid search (rapide)


# ═══════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT — PHASE 1 : grid search sur échantillon
# ═══════════════════════════════════════════════════════════════
def trouver_meilleurs_params(X_tune, y_tune, nom_modele="modele", n_iter=N_ITER_SEARCH):
    """
    Recherche les meilleurs hyperparamètres sur un sous-échantillon (rapide).
    Retourne les meilleurs params sans refit sur tout le dataset.
    """
    param_distributions = {
        "rf__n_estimators":      [200, 300],
        "rf__max_depth":         [10, 15, None],
        "rf__min_samples_split": [5, 10],
        "rf__max_features":      ["sqrt", "log2"],
        "rf__class_weight":      ["balanced", "balanced_subsample"],
    }

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf",    RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_RF,       # ← tous les cœurs sur les arbres
        )),
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring="f1_weighted",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS_SEARCH,       # ← 2 folds en parallèle
        verbose=1,
        refit=False,                # ← pas de refit ici, on le fait sur 50k
    )

    dist = dict(pd.Series(y_tune).value_counts())
    print(f"[Tuning] Distribution échantillon ({len(y_tune)} lignes) — {nom_modele} : {dist}")

    search.fit(X_tune, y_tune)

    best = search.best_params_
    print(f"[Tuning] Meilleurs params ({nom_modele}) :")
    for k, v in best.items():
        print(f"         {k}: {v}")
    print(f"[Tuning] F1 CV (weighted) : {search.best_score_:.4f}\n")

    return best


# ═══════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT — PHASE 2 : refit final sur dataset complet
# ═══════════════════════════════════════════════════════════════
def entrainer_final(X_train, y_train, best_params, nom_modele="modele"):
    """
    Entraîne le pipeline final avec les meilleurs params sur tout le dataset.
    """
    # Extraire les params RF (retirer le préfixe "rf__")
    rf_params = {
        k.replace("rf__", ""): v
        for k, v in best_params.items()
        if k.startswith("rf__")
    }

    dist = dict(pd.Series(y_train).value_counts())
    print(f"[Training] Distribution complète — {nom_modele} : {dist}")

    pipeline_final = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf",    RandomForestClassifier(
            **rf_params,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_RF,
        )),
    ])

    pipeline_final.fit(X_train, y_train)
    print(f"[Training] Refit final terminé sur {len(y_train)} lignes.\n")

    return pipeline_final


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

    rf_model     = model.named_steps["rf"]
    importances  = pd.Series(
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

    # ── Charger les splits encodés ────────────────────────────
    with open(SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)

    X_train     = splits["X_train"]
    X_test      = splits["X_test"]
    y_res_train = splits["y_res_train"]
    y_res_test  = splits["y_res_test"]

    # ── MODÈLE — Resolution ───────────────────────────────────
    print("\n" + "─" * 60)
    print("  MODÈLE : RESOLUTION (Random Forest)")
    print("─" * 60 + "\n")

    t1 = time.time()
    labels_res = list(RESOLUTION_LABELS.values())

    # Phase 1 : grid search rapide sur sous-échantillon
    print(f"[Phase 1] Grid search sur {TUNE_SAMPLE} lignes...\n")
    idx_tune  = np.random.RandomState(RANDOM_STATE).choice(
        len(y_res_train), size=min(TUNE_SAMPLE, len(y_res_train)), replace=False
    )
    X_tune = X_train.iloc[idx_tune]
    y_tune = pd.Series(y_res_train).iloc[idx_tune]

    best_params = trouver_meilleurs_params(
        X_tune, y_tune, nom_modele="Resolution"
    )

    # Phase 2 : refit final sur 100% des données d'entraînement
    print(f"[Phase 2] Refit final sur {len(y_res_train)} lignes...\n")
    model_resolution = entrainer_final(
        X_train, y_res_train, best_params, nom_modele="Resolution"
    )

    # Évaluation
    metrics_res = evaluer_modele(
        model_resolution, X_test, y_res_test,
        nom_modele="Resolution", labels=labels_res,
    )
    afficher_feature_importances(model_resolution, X_train.columns)
    res_ok = verifier_performance(
        metrics_res, "Resolution", seuil_f1=SEUIL_F1_RESOLUTION,
    )

    t_modele = time.time() - t1
    print(f"  Temps modèle Resolution : {t_modele:.1f}s")

    # ── SAUVEGARDE CONDITIONNELLE ─────────────────────────────
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