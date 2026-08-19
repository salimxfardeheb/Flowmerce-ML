import pickle
import time
import threading
from datetime import datetime
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
    OHE_ENCODER,
    SCALER,
    FEATURE_CONTRACT,
    SEUIL_F1_RESOLUTION,
    SEUIL_ACCURACY,
    N_ITER_SEARCH,
    RANDOM_STATE,
    RESOLUTION_LABELS,
)
from src.feature_contract import contrat_depuis_artefacts, ecrire_contrat
from src import reporting as rp

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION MACHINE
#  i5-1135G7 : 4 cœurs physiques / 8 threads logiques
# ─────────────────────────────────────────────────────────────
N_JOBS_LGBM   = -1   # LightGBM utilise tous les cœurs pour construire les arbres
N_JOBS_SEARCH = 1    # 2 folds en parallèle max — évite la surcharge mémoire
TUNE_SAMPLE   = 35000  # lignes pour le grid search


# ═══════════════════════════════════════════════════════════════
#  TIMER TEMPS RÉEL
# ═══════════════════════════════════════════════════════════════
class LiveTimer:
    """
    Affiche le temps écoulé en continu sur le terminal pendant une phase.
    Le rafraîchissement (\\r) ne part que vers la console : le fichier log
    ne reçoit que la ligne finale, une fois la phase terminée.
    """

    def __init__(self, label, flux=None):
        self.label = label
        self.flux = flux if flux is not None else sys.stdout
        self.interactif = bool(getattr(self.flux, "isatty", lambda: False)())
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _ecrire_console(self, texte):
        ecrire = getattr(self.flux, "ecrire_console", self.flux.write)
        ecrire(texte)

    def _run(self):
        while not self._stop.is_set():
            m, s = divmod(int(time.time() - self._t0), 60)
            self._ecrire_console(f"\r  ⏳ {self.label} — {m}m {s:02d}s")
            time.sleep(1)

    def start(self):
        self._t0 = time.time()
        if self.interactif:
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self.interactif:
            self._thread.join()
            self._ecrire_console("\r" + " " * (rp.LARGEUR - 2) + "\r")
        elapsed = time.time() - self._t0
        print(f"  {rp.c('✔', 'vert')} {self.label} — terminé en {rp.fmt_duree(elapsed)}")
        return elapsed


# ═══════════════════════════════════════════════════════════════
#  CAPTURE DE LA SORTIE CONSOLE (tee vers un fichier log)
# ═══════════════════════════════════════════════════════════════
class Tee:
    """
    Duplique stdout vers la console ET un fichier.
    Les codes couleur sont retirés côté fichier : le log reste lisible
    dans un éditeur, la console garde ses couleurs.
    """

    def __init__(self, console, fichier):
        self.console = console
        self.fichier = fichier

    def write(self, data):
        self.console.write(data)
        self.console.flush()
        self.fichier.write(rp.sans_ansi(data))
        self.fichier.flush()

    def ecrire_console(self, data):
        """Sortie éphémère (barres de progression) — jamais journalisée."""
        self.console.write(data)
        self.console.flush()

    def flush(self):
        self.console.flush()
        self.fichier.flush()

    def isatty(self):
        return bool(getattr(self.console, "isatty", lambda: False)())


# ═══════════════════════════════════════════════════════════════
#  APERÇU DES DONNÉES
# ═══════════════════════════════════════════════════════════════
def afficher_donnees(X_train, X_test, y_train, y_test, labels):
    """Volumétrie et répartition des classes, train vs test."""
    rp.sous_section("VOLUMÉTRIE")
    rp.cle_valeur("Lignes d'entraînement", f"{rp.fmt_entier(len(y_train))}")
    rp.cle_valeur("Lignes de test", f"{rp.fmt_entier(len(y_test))}")
    rp.cle_valeur("Features en entrée", f"{rp.fmt_entier(X_train.shape[1])}")

    dist_train = pd.Series(y_train).value_counts()
    dist_test = pd.Series(y_test).value_counts()

    rp.sous_section("RÉPARTITION DES CLASSES")
    print(
        f"  {rp.pad('Classe', 14)}{rp.pad('Train', 10, '>')}{rp.pad('Part', 9, '>')}"
        f"{rp.pad('Test', 10, '>')}{rp.pad('Part', 9, '>')}  Équilibre"
    )
    for i, label in enumerate(labels):
        n_tr = int(dist_train.get(i, 0))
        n_te = int(dist_test.get(i, 0))
        p_tr = n_tr / len(y_train) if len(y_train) else 0.0
        p_te = n_te / len(y_test) if len(y_test) else 0.0
        print(
            f"  {rp.pad(label, 14)}{rp.pad(rp.fmt_entier(n_tr), 10, '>')}"
            f"{rp.pad(f'{p_tr:.1%}', 9, '>')}{rp.pad(rp.fmt_entier(n_te), 10, '>')}"
            f"{rp.pad(f'{p_te:.1%}', 9, '>')}  {rp.barre(p_tr, 16)}"
        )
    return {
        "lignes_train": int(len(y_train)),
        "lignes_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "distribution_train": {labels[i]: int(dist_train.get(i, 0)) for i in range(len(labels))},
        "distribution_test": {labels[i]: int(dist_test.get(i, 0)) for i in range(len(labels))},
    }


# ═══════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT — PHASE 1 : grid search sur échantillon
# ═══════════════════════════════════════════════════════════════
def trouver_meilleurs_params(X_tune, y_tune, nom_modele="modele", n_iter=N_ITER_SEARCH):
    """
    Recherche les meilleurs hyperparamètres sur un sous-échantillon (rapide).
    Retourne (meilleurs params, meilleur score CV) sans refit sur tout le dataset.
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
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("lgbm",  LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_LGBM,
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
        verbose=0,                  # ← la progression est portée par LiveTimer
        refit=False,                # ← pas de refit ici, on le fait sur 50k
    )

    rp.sous_section("PARAMÈTRES DE RECHERCHE")
    rp.cle_valeur("Échantillon de tuning", f"{rp.fmt_entier(len(y_tune))} lignes")
    rp.cle_valeur("Combinaisons testées", f"{n_iter} × 5 folds = {n_iter * 5} fits")
    rp.cle_valeur("Score optimisé", "F1 pondéré (validation croisée)")

    search.fit(X_tune, y_tune)

    best = search.best_params_
    rp.sous_section("HYPERPARAMÈTRES RETENUS")
    for k, v in sorted(best.items()):
        rp.cle_valeur(k.replace("lgbm__", "  "), v)
    print()
    rp.cle_valeur("F1 CV (pondéré)", rp.c(f"{search.best_score_:.4f}", "gras"))

    return best, float(search.best_score_)


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

    rp.sous_section("CONFIGURATION")
    rp.cle_valeur("Lignes d'entraînement", rp.fmt_entier(len(y_train)))
    rp.cle_valeur("Rééquilibrage", "SMOTE (k=5) + class_weight='balanced'")
    print()

    pipeline_final = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
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

    return pipeline_final


# ═══════════════════════════════════════════════════════════════
#  ÉVALUATION
# ═══════════════════════════════════════════════════════════════
def evaluer_modele(model, X_test, y_test, nom_modele="modele", labels=None,
                   seuils=None, f1_cv=None):
    """
    Évalue le modèle et publie le rapport lisible :
    métriques globales, détail par classe, matrice de confusion, diagnostic.
    """
    seuils = seuils or {}
    y_pred = model.predict(X_test)

    metriques = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    rapport, cm = rp.evaluer(y_test, y_pred, labels)

    rp.bloc_metriques(metriques, seuils, len(y_test))
    rp.bloc_par_classe(rapport, labels)
    rp.bloc_matrice(cm, labels)
    constats = rp.bloc_diagnostic(metriques, rapport, labels, cm, f1_cv=f1_cv)

    metriques.update({
        "f1_macro":   rapport["macro avg"]["f1-score"],
        "n_test":     int(len(y_test)),
        "par_classe": {l: rapport[l] for l in labels},
        "matrice":    cm.tolist(),
        "labels":     list(labels),
        "diagnostic": constats,
    })
    return metriques


# ═══════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCES
# ═══════════════════════════════════════════════════════════════
def afficher_feature_importances(model, feature_names, top_n=20):

    lgbm_model  = model.named_steps["lgbm"]
    importances = pd.Series(
        lgbm_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    rp.bloc_importances(importances, top_n=top_n)

    return importances


# ═══════════════════════════════════════════════════════════════
#  DÉCISION DE PERFORMANCE
# ═══════════════════════════════════════════════════════════════
def verifier_performance(metrics, nom_modele, seuil_f1, seuil_acc=SEUIL_ACCURACY):
    """Compare les métriques aux seuils. Retourne (accepté, motifs lisibles)."""
    controles = [
        ("F1 pondéré", metrics["f1"], seuil_f1),
        ("Accuracy",   metrics["accuracy"], seuil_acc),
    ]
    motifs = []
    accepte = True

    rp.sous_section(f"CONTRÔLE DES SEUILS — {nom_modele}")
    for nom, valeur, seuil in controles:
        ok = valeur >= seuil
        accepte = accepte and ok
        etat = rp.c("PASS", "vert") if ok else rp.c("FAIL", "rouge")
        ecart = valeur - seuil
        print(
            f"  {rp.pad(nom, 16)}{rp.pad(f'{valeur:.4f}', 10, '>')}"
            f"   requis ≥ {seuil:.2f}   ({ecart:+.4f})   {etat}"
        )
        motifs.append(
            f"{nom} {valeur:.4f} {'≥' if ok else '<'} {seuil:.2f} requis "
            f"({ecart:+.4f}) — {'PASS' if ok else 'FAIL'}"
        )

    return accepte, motifs


def sauvegarder(objet, nom_fichier):
    joblib.dump(objet, nom_fichier, compress=3)
    print(f"  {rp.c('✔', 'vert')} {nom_fichier}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    t_total = time.time()

    # ── Démarrage de la capture console -> fichier log horodaté ──
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_log = os.path.join(logs_dir, f"training_{horodatage}.txt")
    chemin_json = os.path.join(logs_dir, f"training_{horodatage}.json")

    stdout_original = sys.stdout
    log_file = open(chemin_log, "w", encoding="utf-8")
    sys.stdout = Tee(stdout_original, log_file)
    rp.configurer_couleurs(sys.stdout)

    rp.titre(
        "Rapport d'entraînement — Flowmerce",
        "Modèle    : Resolution (LightGBM + SMOTE)",
        f"Exécution : {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}",
        f"Journal   : logs/training_{horodatage}.txt",
    )

    labels_res = [RESOLUTION_LABELS[k] for k in sorted(RESOLUTION_LABELS)]
    seuils = {"f1": SEUIL_F1_RESOLUTION, "accuracy": SEUIL_ACCURACY}

    # ── 1. Données ────────────────────────────────────────────
    rp.section(1, "Données")

    with open(SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)

    X_train     = splits["X_train"]
    X_test      = splits["X_test"]
    y_res_train = splits["y_res_train"]
    y_res_test  = splits["y_res_test"]

    infos_donnees = afficher_donnees(
        X_train, X_test, y_res_train, y_res_test, labels_res
    )

    # ── 2. Phase 1 : recherche d'hyperparamètres ──────────────
    rp.section(2, "Phase 1 — Recherche d'hyperparamètres")

    idx_tune = np.random.RandomState(RANDOM_STATE).choice(
        len(y_res_train), size=min(TUNE_SAMPLE, len(y_res_train)), replace=False
    )
    X_tune = X_train.iloc[idx_tune]
    y_tune = pd.Series(y_res_train).iloc[idx_tune]

    timer1 = LiveTimer("Phase 1 — Grid search").start()
    best_params, f1_cv = trouver_meilleurs_params(
        X_tune, y_tune, nom_modele="Resolution"
    )
    t_phase1 = timer1.stop()

    # ── 3. Phase 2 : refit final ──────────────────────────────
    rp.section(3, "Phase 2 — Entraînement final")

    timer2 = LiveTimer("Phase 2 — Refit final").start()
    model_resolution = entrainer_final(
        X_train, y_res_train, best_params, nom_modele="Resolution"
    )
    t_phase2 = timer2.stop()

    # ── 4. Évaluation ─────────────────────────────────────────
    rp.section(4, "Évaluation sur le jeu de test")

    t_eval = time.time()
    metrics_res = evaluer_modele(
        model_resolution, X_test, y_res_test,
        nom_modele="Resolution", labels=labels_res,
        seuils=seuils, f1_cv=f1_cv,
    )

    # ── 5. Interprétabilité ───────────────────────────────────
    rp.section(5, "Interprétabilité")
    importances = afficher_feature_importances(model_resolution, X_train.columns)
    t_eval = time.time() - t_eval

    # ── 6. Décision et sauvegarde ─────────────────────────────
    rp.section(6, "Décision et sauvegarde")

    res_ok, motifs = verifier_performance(
        metrics_res, "Resolution", seuil_f1=SEUIL_F1_RESOLUTION,
    )

    t_save = time.time()
    artefacts = []

    if res_ok:
        rp.sous_section("ARTEFACTS ÉCRITS")
        sauvegarder(model_resolution, MODEL_RESOLUTION)
        sauvegarder(list(X_train.columns), TRAIN_COLUMNS)
        artefacts += [str(MODEL_RESOLUTION), str(TRAIN_COLUMNS)]

        # Contrat de features — dérivé du jeu d'artefacts qui vient d'être figé.
        # C'est ici, et nulle part ailleurs, que le vocabulaire servi est défini :
        # il ne peut donc pas diverger de l'encodeur réellement déployé (C-02).
        contrat = contrat_depuis_artefacts(
            joblib.load(OHE_ENCODER),
            joblib.load(SCALER),
            list(X_train.columns),
            RESOLUTION_LABELS,
        )
        ecrire_contrat(contrat)
        print(f"  {rp.c('✔', 'vert')} {FEATURE_CONTRACT}  "
              f"(version {contrat['contract_version']})")
        artefacts.append(f"{FEATURE_CONTRACT} (v{contrat['contract_version']})")
    else:
        rp.sous_section("ARTEFACTS")
        print(f"  {rp.c('✖', 'rouge')} Aucun fichier écrit — les seuils ne sont pas atteints.")

    t_save = time.time() - t_save
    elapsed = time.time() - t_total

    rp.bloc_durees(
        [
            ("Phase 1 — Grid search",     t_phase1),
            ("Phase 2 — Refit final",     t_phase2),
            ("Évaluation + importances",  t_eval),
            ("Sauvegarde des artefacts",  t_save),
            ("Chargement + divers",       max(0.0, elapsed - t_phase1 - t_phase2 - t_eval - t_save)),
        ],
        elapsed,
    )

    rp.bloc_bilan(res_ok, motifs, artefacts, elapsed)

    if res_ok:
        print()
        rp.puce("Prochaine étape — répercuter le contrat côté web app :", "jaune")
        rp.puce("  cp contracts/feature_contract.json "
                "../flowmerce-web-app/lib/ml/feature-contract.json",
                "dim", replier=False)
    else:
        print()
        rp.puce("Pistes — élargir N_ITER_SEARCH / TUNE_SAMPLE, revoir les features "
                "des classes en échec, ou ajuster les seuils dans config.py.", "jaune")

    # ── Rapport machine, pour comparer les runs entre eux ──────
    rapport_json = rp.construire_rapport(
        horodatage=horodatage,
        date=datetime.now().isoformat(timespec="seconds"),
        modele="Resolution",
        algorithme="LightGBM + SMOTE",
        donnees=infos_donnees,
        hyperparametres=best_params,
        f1_cv=f1_cv,
        seuils={"f1": SEUIL_F1_RESOLUTION, "accuracy": SEUIL_ACCURACY},
        metriques={k: v for k, v in metrics_res.items() if k != "diagnostic"},
        diagnostic=metrics_res["diagnostic"],
        top_features=importances.head(20).to_dict(),
        durees={
            "phase1_grid_search": t_phase1,
            "phase2_refit": t_phase2,
            "evaluation": t_eval,
            "sauvegarde": t_save,
            "total": elapsed,
        },
        accepte=res_ok,
        artefacts=artefacts,
    )
    rp.ecrire_rapport_json(chemin_json, rapport_json)
    print()
    rp.puce(f"Rapport texte : logs/training_{horodatage}.txt", "dim")
    rp.puce(f"Rapport JSON  : logs/training_{horodatage}.json", "dim")

    # ── Fin de la capture console ──────────────────────────────
    sys.stdout = stdout_original
    log_file.close()
