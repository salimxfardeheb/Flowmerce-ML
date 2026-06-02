import pickle
import time
import threading
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
N_JOBS_LGBM   = -1   # LightGBM utilise tous les cœurs pour construire les arbres
N_JOBS_SEARCH = 2    # 2 folds en parallèle max — évite la surcharge mémoire
TUNE_SAMPLE   = 30000  # lignes pour le grid search (rapide)


# ═══════════════════════════════════════════════════════════════
#  TIMER TEMPS RÉEL
# ═══════════════════════════════════════════════════════════════
class LiveTimer:
    """Affiche le temps écoulé en continu sur le terminal pendant une phase."""

    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            elapsed = time.time() - self._t0
            m, s = divmod(int(elapsed), 60)
            print(f"\r  [TIMER] {self.label} — {m}m {s:02d}s", end="", flush=True)
            time.sleep(1)

    def start(self):
        self._t0 = time.time()
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        elapsed = time.time() - self._t0
        m, s = divmod(int(elapsed), 60)
        print(f"\r  [OK]    {self.label} — {m}m {s:02d}s", flush=True)
        return elapsed


# ═══════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT — PHASE 1 : grid search sur échantillon
# ═══════════════════════════════════════════════════════════════
def trouver_meilleurs_params(X_tune, y_tune, nom_modele="modele", n_iter=N_ITER_SEARCH):
    """
    Recherche les meilleurs hyperparamètres sur un sous-échantillon (rapide).
    Retourne les meilleurs params sans refit sur tout le dataset.
    """
    param_distributions = {
        "lgbm__n_estimators":      [200, 400, 600],
        "lgbm__num_leaves":        [31, 63, 127],
        "lgbm__max_depth":         [-1, 12, 20],
        "lgbm__learning_rate":     [0.02, 0.05, 0.1],
        "lgbm__min_child_samples": [10, 20, 50],
        "lgbm__subsample":         [0.8, 1.0],
        "lgbm__colsample_bytree":  [0.8, 1.0],
    }

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("lgbm",  LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",   # ← aide les classes rares (Refund)
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_LGBM,        # ← tous les cœurs sur les arbres
            verbose=-1,
        )),
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=5,
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
    # Extraire les params LightGBM (retirer le préfixe "lgbm__")
    lgbm_params = {
        k.replace("lgbm__", ""): v
        for k, v in best_params.items()
        if k.startswith("lgbm__")
    }

    dist = dict(pd.Series(y_train).value_counts())
    print(f"[Training] Distribution complète — {nom_modele} : {dist}")

    pipeline_final = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("lgbm",  LGBMClassifier(
            **lgbm_params,
            objective="multiclass",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_LGBM,
            verbose=-1,
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

    lgbm_model   = model.named_steps["lgbm"]
    importances  = pd.Series(
        lgbm_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    # LightGBM renvoie un nombre de splits (entiers) — on normalise pour l'affichage
    total = importances.sum()
    importances_norm = importances / total if total else importances

    print(f"\n{'─' * 50}")
    print(f"  TOP {top_n} FEATURE IMPORTANCES")
    print(f"{'─' * 50}")
    for i, feat in enumerate(importances.head(top_n).index, 1):
        imp = importances_norm[feat]
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
    joblib.dump(objet, nom_fichier, compress=3)
    print(f"[Sauvegarde] {nom_fichier}")


# ═══════════════════════════════════════════════════════════════
#  ENREGISTREMENT DES RÉSULTATS (log horodaté)
# ═══════════════════════════════════════════════════════════════
def sauvegarder_rapport(metrics, y_test, y_pred, labels, temps_phases, logs_dir):
    """Sauvegarde métriques + classification report dans un .txt horodaté."""
    from datetime import datetime

    os.makedirs(logs_dir, exist_ok=True)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin = os.path.join(logs_dir, f"training_{horodatage}.txt")

    with open(chemin, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  RAPPORT D'ENTRAINEMENT — {horodatage}\n")
        f.write("=" * 60 + "\n\n")

        f.write("METRIQUES\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Accuracy  : {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision : {metrics['precision']:.4f}\n")
        f.write(f"  Recall    : {metrics['recall']:.4f}\n")
        f.write(f"  F1-score  : {metrics['f1']:.4f}\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
        f.write("\n")

        f.write("MATRICE DE CONFUSION\n")
        f.write("-" * 40 + "\n")
        cm = confusion_matrix(y_test, y_pred)
        f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
        f.write("\n\n")

        f.write("TEMPS D'EXECUTION\n")
        f.write("-" * 40 + "\n")
        for label, duree in temps_phases.items():
            m, s = divmod(int(duree), 60)
            f.write(f"  {label:<25s} : {m}m {s:02d}s\n")
        f.write("\n")

    print(f"[Rapport] Sauvegarde -> {chemin}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    t_total = time.time()

    print("=" * 60)
    print("  PIPELINE D'ENTRAÎNEMENT — FLOWMERCE (LightGBM)")
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
    print("  MODÈLE : RESOLUTION (LightGBM)")
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

    timer1 = LiveTimer("Phase 1 — Grid search").start()
    best_params = trouver_meilleurs_params(
        X_tune, y_tune, nom_modele="Resolution"
    )
    t_phase1 = timer1.stop()

    # Phase 2 : refit final sur 100% des données d'entraînement
    print(f"[Phase 2] Refit final sur {len(y_res_train)} lignes...\n")
    timer2 = LiveTimer("Phase 2 — Refit final").start()
    model_resolution = entrainer_final(
        X_train, y_res_train, best_params, nom_modele="Resolution"
    )
    t_phase2 = timer2.stop()

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

    # Log horodaté des résultats
    sauvegarder_rapport(
        metrics=metrics_res,
        y_test=y_res_test,
        y_pred=model_resolution.predict(X_test),
        labels=labels_res,
        temps_phases={
            "Phase 1 - Grid search": t_phase1,
            "Phase 2 - Refit final": t_phase2,
            "Total modele":          t_modele,
        },
        logs_dir=os.path.join(os.path.dirname(__file__), "..", "logs"),
    )

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