"""
Mise en forme des rapports d'entraînement (console + fichier log).

Un seul endroit décide de la présentation : src/training.py se contente
d'appeler ces blocs. La couleur n'est émise que sur un vrai terminal, et
le Tee la retire du fichier log — les deux sorties restent alignées.
"""

import json
import os
import re
import sys
import textwrap

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

LARGEUR = 74

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CODES = {
    "gras":  "\x1b[1m",
    "dim":   "\x1b[2m",
    "rouge": "\x1b[31m",
    "vert":  "\x1b[32m",
    "jaune": "\x1b[33m",
    "cyan":  "\x1b[36m",
}
_RESET = "\x1b[0m"
_couleurs = False


# ═══════════════════════════════════════════════════════════════
#  PRIMITIVES
# ═══════════════════════════════════════════════════════════════
def configurer_couleurs(flux=None):
    """Active la couleur uniquement si la sortie est un vrai terminal."""
    global _couleurs
    flux = flux if flux is not None else sys.stdout
    if os.environ.get("NO_COLOR"):
        _couleurs = False
    else:
        _couleurs = bool(getattr(flux, "isatty", lambda: False)())
    return _couleurs


def sans_ansi(texte):
    return _ANSI.sub("", texte)


def c(texte, *styles):
    """Colore un fragment. Sans terminal, renvoie le texte tel quel."""
    if not _couleurs or not styles:
        return texte
    return "".join(_CODES[s] for s in styles) + texte + _RESET


def pad(texte, n, align="<"):
    """Complète à n colonnes en ignorant les codes couleur."""
    manque = " " * max(0, n - len(sans_ansi(texte)))
    return texte + manque if align == "<" else manque + texte


def fmt_entier(n):
    return f"{int(n):,}".replace(",", " ")


def fmt_duree(secondes):
    m, s = divmod(int(secondes), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def guillemets(texte):
    """Guillemets français à espaces insécables — le nom ne sera pas coupé."""
    return "\u00ab\u00a0" + str(texte) + "\u00a0\u00bb"


def tronquer(texte, n):
    texte = str(texte)
    return texte if len(texte) <= n else texte[: n - 1] + "…"


def habiller(texte, indent=4, retrait=6):
    """Découpe un texte long pour qu'il tienne dans la largeur du rapport."""
    lignes = textwrap.wrap(texte, width=LARGEUR - indent) or [""]
    suite = textwrap.wrap(" ".join(lignes[1:]), width=LARGEUR - retrait) if lignes[1:] else []
    return [lignes[0]] + suite


def barre(valeur, largeur=20, maxi=1.0):
    """Jauge pleine/vide proportionnelle à `valeur` sur `maxi`."""
    if maxi <= 0:
        return "░" * largeur
    plein = int(round(max(0.0, min(valeur / maxi, 1.0)) * largeur))
    return "█" * plein + "░" * (largeur - plein)


# ═══════════════════════════════════════════════════════════════
#  STRUCTURE DU DOCUMENT
# ═══════════════════════════════════════════════════════════════
def titre(texte, *lignes_info):
    interieur = LARGEUR - 2
    print("╔" + "═" * interieur + "╗")
    print("║ " + pad(c(texte.upper(), "gras"), interieur - 2) + " ║")
    for info in lignes_info:
        print("║ " + pad(c(info, "dim"), interieur - 2) + " ║")
    print("╚" + "═" * interieur + "╝")


def section(numero, texte):
    entete = f"━━━ {numero} · {texte.upper()} "
    print()
    print(c(entete + "━" * max(0, LARGEUR - len(entete)), "gras", "cyan"))


def sous_section(texte):
    print()
    print("  " + c(texte, "gras"))
    print("  " + "─" * (LARGEUR - 4))


def filet():
    print("  " + "─" * (LARGEUR - 4))


def puce(texte, style=None, symbole=None, style_symbole=None, replier=True):
    """Élément de liste, replié sur la largeur du rapport si besoin."""
    prefixe = f"{c(symbole, style_symbole) if style_symbole else symbole} " if symbole else ""
    retrait = 6 if symbole else 4
    lignes = habiller(texte, retrait=retrait) if replier else [texte]
    for i, ligne in enumerate(lignes):
        corps = c(ligne, style) if style else ligne
        print(("    " + prefixe + corps) if i == 0 else (" " * retrait + corps))


def cle_valeur(cle, valeur, largeur_cle=28):
    print(f"  {pad(cle, largeur_cle)}{valeur}")


# ═══════════════════════════════════════════════════════════════
#  BLOC — MÉTRIQUES GLOBALES
# ═══════════════════════════════════════════════════════════════
def _statut(valeur, seuil):
    if seuil is None:
        return "—", None, "—"
    ecart = valeur - seuil
    ok = ecart >= 0
    return (
        c("PASS" if ok else "FAIL", "vert" if ok else "rouge"),
        ok,
        f"{ecart:+.4f}",
    )


def bloc_metriques(metriques, seuils, n_test):
    """Tableau des 4 métriques globales, comparées à leurs seuils."""
    sous_section(f"MÉTRIQUES GLOBALES — {fmt_entier(n_test)} lignes de test")
    print(
        f"  {pad('Métrique', 16)}{pad('Valeur', 10, '>')}"
        f"{pad('Seuil', 12, '>')}{pad('Écart', 11, '>')}  {pad('Statut', 8)}Jauge"
    )
    lignes = [
        ("Accuracy",       metriques["accuracy"],  seuils.get("accuracy")),
        ("F1 (pondéré)",   metriques["f1"],        seuils.get("f1")),
        ("Précision (p.)", metriques["precision"], None),
        ("Rappel (p.)",    metriques["recall"],    None),
    ]
    for nom, valeur, seuil in lignes:
        libelle, _, ecart = _statut(valeur, seuil)
        texte_seuil = f"≥ {seuil:.2f}" if seuil is not None else "—"
        print(
            f"  {pad(nom, 16)}{pad(f'{valeur:.4f}', 10, '>')}"
            f"{pad(texte_seuil, 12, '>')}{pad(ecart, 11, '>')}  "
            f"{pad(libelle, 8)}{barre(valeur, 12)}"
        )


# ═══════════════════════════════════════════════════════════════
#  BLOC — DÉTAIL PAR CLASSE
# ═══════════════════════════════════════════════════════════════
def _qualite(f1, support):
    if support == 0:
        return c("○ absente", "dim")
    if f1 == 0:
        return c("✖ ignorée", "rouge")
    if f1 >= 0.80:
        return c("● bon", "vert")
    if f1 >= 0.65:
        return c("● correct", "jaune")
    if f1 >= 0.50:
        return c("● faible", "jaune")
    return c("● critique", "rouge")


def bloc_par_classe(rapport, labels):
    """Précision / rappel / F1 par classe, avec un verdict lisible."""
    total = sum(rapport[l]["support"] for l in labels)
    sous_section("DÉTAIL PAR CLASSE")
    print(
        f"  {pad('Classe', 18)}{pad('Support', 8, '>')}{pad('Part', 7, '>')}"
        f"{pad('Précis.', 9, '>')}{pad('Rappel', 8, '>')}{pad('F1', 7, '>')}  Qualité"
    )
    for label in labels:
        r = rapport[label]
        part = r["support"] / total * 100 if total else 0.0
        prec = format(r["precision"], ".2f")
        rapp = format(r["recall"], ".2f")
        f1 = format(r["f1-score"], ".2f")
        print(
            f"  {pad(tronquer(label, 17), 18)}{pad(fmt_entier(r['support']), 8, '>')}"
            f"{pad(f'{part:.1f}%', 7, '>')}{pad(prec, 9, '>')}"
            f"{pad(rapp, 8, '>')}{pad(f1, 7, '>')}  "
            f"{_qualite(r['f1-score'], r['support'])}"
        )
    filet()
    for cle, nom in (("macro avg", "Moyenne macro"), ("weighted avg", "Moyenne pondérée")):
        r = rapport[cle]
        prec = format(r["precision"], ".2f")
        rapp = format(r["recall"], ".2f")
        f1 = format(r["f1-score"], ".2f")
        print(
            f"  {pad(nom, 18)}{pad('', 8)}{pad('', 7)}"
            f"{pad(prec, 9, '>')}{pad(rapp, 8, '>')}{pad(f1, 7, '>')}"
        )
    print()
    puce("Précision = parmi les prédictions de cette classe, part de justes.", "dim")
    puce("Rappel    = parmi les cas réels de cette classe, part de retrouvés.", "dim")


# ═══════════════════════════════════════════════════════════════
#  BLOC — MATRICE DE CONFUSION
# ═══════════════════════════════════════════════════════════════
def bloc_matrice(cm, labels):
    """Effectifs + pourcentage par ligne, puis les confusions dominantes."""
    sous_section("MATRICE DE CONFUSION — ligne = réel, colonne = prédit")
    intitule = "réel \\ prédit"
    largeur_label = max(len(intitule) + 2, max(len(l) for l in labels) + 2)

    # Effectif + % par ligne si la largeur le permet, sinon % seul :
    # au-delà de 3 classes, le tableau détaillé déborderait.
    totaux = [int(cm[i].sum()) for i in range(len(labels))]
    detail = [
        [f"{fmt_entier(int(cm[i][j]))} {(cm[i][j] / totaux[i] * 100 if totaux[i] else 0):5.1f}%"
         for j in range(len(labels))]
        for i in range(len(labels))
    ]
    largeur_cell = max(max(len(x) for ligne in detail for x in ligne),
                       max(len(l) for l in labels)) + 2
    complet = 2 + largeur_label + largeur_cell * len(labels) + 9 <= LARGEUR
    if not complet:
        detail = [
            [f"{(cm[i][j] / totaux[i] * 100 if totaux[i] else 0):5.1f}%"
             for j in range(len(labels))]
            for i in range(len(labels))
        ]
        largeur_cell = max(8, max(len(l) for l in labels) + 2)

    entete = "  " + pad(intitule, largeur_label)
    for label in labels:
        entete += pad(label, largeur_cell, ">")
    print(entete + pad("Total", 9, ">"))

    for i, reel in enumerate(labels):
        ligne = "  " + pad(reel, largeur_label)
        for j in range(len(labels)):
            pct = cm[i][j] / totaux[i] * 100 if totaux[i] else 0.0
            cellule = detail[i][j]
            if i == j:
                cellule = c(cellule, "vert" if pct >= 70 else "jaune")
            elif pct >= 20:
                cellule = c(cellule, "rouge")
            ligne += pad(cellule, largeur_cell, ">")
        print(ligne + pad(fmt_entier(totaux[i]), 9, ">"))
    if not complet:
        print()
        puce("Cellules en % de la ligne — effectifs détaillés dans le rapport JSON.", "dim")

    confusions = []
    for i, reel in enumerate(labels):
        total_ligne = int(cm[i].sum())
        for j, predit in enumerate(labels):
            if i != j and cm[i][j] > 0:
                confusions.append(
                    (int(cm[i][j]), reel, predit, cm[i][j] / total_ligne * 100 if total_ligne else 0.0)
                )
    if confusions:
        print()
        print("  " + c("Confusions dominantes", "gras"))
        for n, reel, predit, pct in sorted(confusions, reverse=True)[:5]:
            puce(
                f"{pad(tronquer(reel, 13), 14)}→ {pad(tronquer(predit, 13), 14)}"
                f"{pad(fmt_entier(n), 7, '>')} cas   ({pct:.1f}% des {reel} réels)",
                replier=False,
            )


# ═══════════════════════════════════════════════════════════════
#  BLOC — DIAGNOSTIC AUTOMATIQUE
# ═══════════════════════════════════════════════════════════════
def bloc_diagnostic(metriques, rapport, labels, cm, f1_cv=None):
    """
    Traduit les chiffres en constats actionnables : classes ignorées,
    rappel faible, écart CV/test, gain réel face au réflexe majoritaire.
    """
    sous_section("DIAGNOSTIC")
    constats = []

    supports = {l: rapport[l]["support"] for l in labels}
    total = sum(supports.values())

    # Référence naïve : toujours prédire la classe la plus fréquente.
    majoritaire = max(supports, key=supports.get) if total else None
    if majoritaire:
        base = supports[majoritaire] / total
        gain = metriques["accuracy"] - base
        style = "vert" if gain >= 0.10 else ("jaune" if gain > 0.02 else "rouge")
        constats.append((
            style,
            f"Référence naïve (toujours {guillemets(majoritaire)}) : {base:.1%} "
            f"— le modèle apporte {gain:+.1%} d'accuracy.",
        ))

    for label in labels:
        r = rapport[label]
        if r["support"] == 0:
            continue
        if r["f1-score"] == 0:
            constats.append((
                "rouge",
                f"{guillemets(label)} n'est jamais prédite "
                f"({fmt_entier(r['support'])} cas réels perdus) "
                f"— classe trop rare ou features non discriminantes.",
            ))
        elif r["recall"] < 0.60:
            i = labels.index(label)
            j = int(np.argmax([cm[i][k] if k != i else -1 for k in range(len(labels))]))
            constats.append((
                "jaune",
                f"{guillemets(label)} : rappel {r['recall']:.0%} — "
                f"{1 - r['recall']:.0%} des cas manqués, "
                f"surtout classés {guillemets(labels[j])}.",
            ))
        elif r["precision"] < 0.60:
            constats.append((
                "jaune",
                f"{guillemets(label)} : précision {r['precision']:.0%} — "
                f"{1 - r['precision']:.0%} des prédictions sont des faux positifs.",
            ))

    if total:
        part_max = max(supports.values()) / total
        part_min = min(supports.values()) / total
        if part_min and part_max / part_min >= 3:
            constats.append((
                "jaune",
                f"Déséquilibre des classes {part_max:.0%} / {part_min:.0%} "
                f"— privilégier le F1 macro à l'accuracy pour juger le modèle.",
            ))

    if f1_cv is not None:
        ecart = f1_cv - metriques["f1"]
        if ecart > 0.05:
            constats.append((
                "jaune",
                f"F1 CV {f1_cv:.4f} vs F1 test {metriques['f1']:.4f} ({ecart:+.4f}) "
                f"— signe de surapprentissage sur l'échantillon de tuning.",
            ))
        else:
            constats.append((
                "vert",
                f"F1 CV {f1_cv:.4f} vs F1 test {metriques['f1']:.4f} ({-ecart:+.4f}) "
                f"— généralisation cohérente.",
            ))

    if not constats:
        constats.append(("vert", "Aucun point d'attention détecté."))

    for style, texte in constats:
        symbole = {"vert": "✔", "jaune": "▲", "rouge": "✖"}[style]
        puce(texte, symbole=symbole, style_symbole=style)

    return [texte for _, texte in constats]


# ═══════════════════════════════════════════════════════════════
#  BLOC — FEATURE IMPORTANCES
# ═══════════════════════════════════════════════════════════════
def bloc_importances(importances, top_n=20):
    """Top features avec part relative et cumul — où le modèle regarde."""
    total = importances.sum()
    sous_section(f"TOP {top_n} FEATURES — poids dans les décisions de l'arbre")
    tete = importances.head(top_n)
    maxi = tete.iloc[0] / total if total else 1.0
    cumul = 0.0
    for i, (feat, valeur) in enumerate(tete.items(), 1):
        part = valeur / total if total else 0.0
        cumul += part
        print(
            f"  {i:2d}. {pad(tronquer(feat, 27), 28)}{pad(f'{part:.2%}', 8, '>')}  "
            f"{barre(part, 16, maxi)}  {c(f'cumul {cumul:5.1%}', 'dim')}"
        )
    reste = 1.0 - cumul
    print()
    puce(
        f"Les {top_n} premières features portent {cumul:.1%} du signal "
        f"({len(importances)} features au total, {reste:.1%} pour le reste).",
        "dim",
    )


# ═══════════════════════════════════════════════════════════════
#  BLOC — TEMPS D'EXÉCUTION
# ═══════════════════════════════════════════════════════════════
def bloc_durees(phases, total):
    """phases : liste de (libellé, secondes)."""
    sous_section("TEMPS D'EXÉCUTION")
    print(f"  {pad('Phase', 32)}{pad('Durée', 12, '>')}{pad('Part', 9, '>')}  Répartition")
    for libelle, secondes in phases:
        part = secondes / total if total else 0.0
        print(
            f"  {pad(tronquer(libelle, 31), 32)}{pad(fmt_duree(secondes), 12, '>')}"
            f"{pad(f'{part:.1%}', 9, '>')}  {barre(part, 16)}"
        )
    filet()
    print(f"  {pad('Total', 32)}{pad(fmt_duree(total), 12, '>')}{pad('100.0%', 9, '>')}")


# ═══════════════════════════════════════════════════════════════
#  BLOC — BILAN FINAL
# ═══════════════════════════════════════════════════════════════
def bloc_bilan(accepte, motifs, artefacts, duree_totale):
    """Encadré de fin : verdict, motifs, artefacts écrits."""
    interieur = LARGEUR - 2
    verdict = (
        c("MODÈLE RETENU — artefacts sauvegardés", "gras", "vert")
        if accepte
        else c("MODÈLE REJETÉ — aucun artefact écrasé", "gras", "rouge")
    )
    print()
    print("╔" + "═" * interieur + "╗")
    print("║ " + pad(verdict, interieur - 2) + " ║")
    print("╟" + "─" * interieur + "╢")
    for motif in motifs:
        print("║ " + pad(f"  {motif}", interieur - 2) + " ║")
    if artefacts:
        print("║ " + pad("", interieur - 2) + " ║")
        for artefact in artefacts:
            print("║ " + pad(f"  → {artefact}", interieur - 2) + " ║")
    print("╟" + "─" * interieur + "╢")
    print("║ " + pad(f"  Durée totale : {fmt_duree(duree_totale)}", interieur - 2) + " ║")
    print("╚" + "═" * interieur + "╝")


# ═══════════════════════════════════════════════════════════════
#  RAPPORT MACHINE (JSON)
# ═══════════════════════════════════════════════════════════════
def construire_rapport(**donnees):
    """Normalise en types JSON-sérialisables (numpy → python)."""
    def convertir(o):
        if isinstance(o, dict):
            return {str(k): convertir(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convertir(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return convertir(o.tolist())
        if isinstance(o, pd.Series):
            return convertir(o.to_dict())
        return o

    return convertir(donnees)


def ecrire_rapport_json(chemin, rapport):
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)
    return chemin


# ═══════════════════════════════════════════════════════════════
#  CALCUL DES ÉLÉMENTS D'ÉVALUATION
# ═══════════════════════════════════════════════════════════════
def evaluer(y_vrai, y_pred, labels):
    """Rapport par classe + matrice de confusion, indexés par nom de classe."""
    indices = list(range(len(labels)))
    rapport = classification_report(
        y_vrai, y_pred, labels=indices, target_names=labels,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_vrai, y_pred, labels=indices)
    return rapport, cm
