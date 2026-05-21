"""
fuzzy_evaluation.py — Méthode Average Score Fusion (ASF)
───────────────────────────────────────────────────────────────────────────────
Implémente l'évaluation Fuzzy AHP avec fusion ASF :

  Γ(xp) = Σ_D  w̄_D · Φ_D(xp)          ← score individuel (Eq. 1)
  Γ*_L  = (1/|G_L|) Σ_{p∈G_L} Γ(xp)   ← moyenne de groupe (Eq. 2)

Le label GMM (Beginner/Intermediate/Expert) est annotatif uniquement.
Aucun PPIF ni facteur d'ajustement n'est appliqué au score numérique.

Lit  : db.players_classified, db.model_metrics
Écrit: db.fuzzy_results, db.fuzzy_evaluation, db.fuzzy_weights_cache
"""

import math
from typing import Dict, List
import numpy as np
from pymongo import MongoClient

from fuzzy_models import (
    TFN, DIMENSIONS, Dimension, Criterion,
    CriterionResult, DimensionResult, EvaluationResult,
)
from fuzzy_ahp import FuzzyAHP


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS TFN
# ─────────────────────────────────────────────────────────────────────────────

def _t(l, m, u) -> TFN:
    return TFN(float(l), float(m), float(u))

def _r(l, m, u) -> TFN:
    return TFN(1.0 / u, 1.0 / m, 1.0 / l)


# ─────────────────────────────────────────────────────────────────────────────
# MATRICES DE COMPARAISON (inchangées)
# ─────────────────────────────────────────────────────────────────────────────

EQ  = _t(1,1,1); WK  = _t(1,2,3); MOD = _t(2,3,4)
STR = _t(3,4,5); HI  = _t(4,5,6); VH  = _t(6,7,8); EXT = _t(8,9,9)

DIMENSION_UPPER: List[TFN] = [MOD, HI, VH, MOD, STR, WK]
PD_UPPER: List[TFN] = [WK, MOD, STR, WK, MOD, WK]
TD_UPPER: List[TFN] = [WK, MOD, MOD, WK, WK, WK]
LD_UPPER: List[TFN] = [WK, MOD, MOD, WK, WK, WK]
BD_UPPER: List[TFN] = [WK, MOD, WK]

CRITERIA_UPPER_TRIANGLES: Dict[str, List[TFN]] = {
    "PD": PD_UPPER, "TD": TD_UPPER, "LD": LD_UPPER, "BD": BD_UPPER,
}


# ─────────────────────────────────────────────────────────────────────────────
# POIDS FUZZY AHP (calculés une fois au démarrage)
# ─────────────────────────────────────────────────────────────────────────────

class AHPWeights:
    def __init__(self):
        self.ahp = FuzzyAHP()

        print("\n═══════════════════════════════════════════")
        print("  Computing Fuzzy AHP Weights  [ASF mode]")
        print("═══════════════════════════════════════════")

        print("\n▸ Dimension-level matrix (PD, TD, LD, BD)")
        dim_result = self.ahp.run(DIMENSION_UPPER, n=4)
        self.dim_cr:             float       = dim_result["cr"]
        self.dim_fuzzy_weights:  List[TFN]   = dim_result["fuzzy_weights"]
        self.dim_weights:        List[float] = dim_result["normalized_weights"]

        self.crit_cr:            Dict[str, float]       = {}
        self.crit_fuzzy_weights: Dict[str, List[TFN]]   = {}
        self.crit_weights:       Dict[str, List[float]] = {}

        for dim in DIMENSIONS:
            upper = CRITERIA_UPPER_TRIANGLES[dim.code]
            nc    = len(dim.criteria)
            print(f"\n▸ {dim.code} criteria ({', '.join(c.code for c in dim.criteria)})")
            res = self.ahp.run(upper, n=nc)
            self.crit_cr[dim.code]            = res["cr"]
            self.crit_fuzzy_weights[dim.code] = res["fuzzy_weights"]
            self.crit_weights[dim.code]       = res["normalized_weights"]

        print("\n═══════════════════════════════════════════")
        for dim, w in zip(DIMENSIONS, self.dim_weights):
            print(f"    {dim.code}: {w:.4f}")
        print("═══════════════════════════════════════════\n")

    def to_dict(self) -> dict:
        out = {"dimension_cr": round(self.dim_cr, 4), "dimensions": []}
        for i, dim in enumerate(DIMENSIONS):
            d_entry = {
                "code":       dim.code,
                "name":       dim.name,
                "weight":     round(self.dim_weights[i], 4),
                "weight_tfn": self.dim_fuzzy_weights[i].to_dict(),
                "cr":         round(self.crit_cr[dim.code], 4),
                "criteria":   [],
            }
            for j, crit in enumerate(dim.criteria):
                d_entry["criteria"].append({
                    "code":       crit.code,
                    "name":       crit.name,
                    "weight":     round(self.crit_weights[dim.code][j], 4),
                    "weight_tfn": self.crit_fuzzy_weights[dim.code][j].to_dict(),
                })
            out["dimensions"].append(d_entry)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS DE SCORE DES CRITÈRES
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

# PD [Lr, Ts, Pc, Em]
def score_Lr(p): return _clamp(p.get("score", 0) / 100.0)
def score_Ts(p): return _clamp(p.get("level", 1) / 10.0)
def score_Pc(p):
    raw = p.get("success", 0)
    return _clamp(float(raw) if raw <= 1 else raw / 100.0)
def score_Em(p): return _clamp(1.0 - p.get("errors", 0) / 10.0)

# TD [Ui, P, U, Gd]
def score_Ui(p):
    clicks = p.get("clicks", 0)
    return _clamp(math.exp(-0.5 * ((clicks - 50.0) / 60.0) ** 2))
def score_P(p):  return _clamp(1.0 - p.get("response_time", 1.0) / 5.0)
def score_U(p):  return _clamp(1.0 - p.get("time", 60) / 120.0)
def score_Gd(p): return _clamp(p.get("moves", 0) / 50.0)

# LD [C, G, F, I]
def score_C(p):
    s = p.get("score", 0) / 100.0
    e = _clamp(p.get("errors", 0) / 10.0)
    return _clamp(s * (1 - abs(e - 0.3) / 0.7))
def score_G(p):
    return _clamp(p.get("moves", 0) / max(p.get("clicks", 1), 1) / 2.0)
def score_F(p):  return _clamp(p.get("repetition", 0) / 5.0)
def score_I(p):
    return (_clamp(p.get("time", 0) / 60.0) + _clamp(p.get("repetition", 0) / 5.0)) / 2.0

# BD [M, E, Ue]
def score_M(p):  return _clamp(p.get("repetition", 0) / 5.0)
def score_E(p):  return _clamp((p.get("clicks", 0) + p.get("moves", 0)) / 100.0)
def score_Ue(p):
    return (p.get("score", 0) / 100.0 + _clamp(1.0 - p.get("errors", 0) / 10.0)) / 2.0

CRITERION_SCORERS = {
    "Lr": score_Lr, "Ts": score_Ts, "Pc": score_Pc, "Em": score_Em,
    "Ui": score_Ui, "P":  score_P,  "U":  score_U,  "Gd": score_Gd,
    "C":  score_C,  "G":  score_G,  "F":  score_F,  "I":  score_I,
    "M":  score_M,  "E":  score_E,  "Ue": score_Ue,
}


# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATEUR ASF
# ─────────────────────────────────────────────────────────────────────────────

class PlayerEvaluator:
    """
    Évalue chaque joueur selon la méthode ASF.
    Le label GMM est stocké comme annotation — il ne modifie pas le score.
    """

    def __init__(self, weights: AHPWeights):
        self.weights = weights

    def evaluate(self, player: dict) -> EvaluationResult:
        dim_results: List[DimensionResult] = []
        # Γ(xp) = Σ_D  w̄_D · Φ_D(xp)
        gamma = 0.0

        for d_idx, dim in enumerate(DIMENSIONS):
            dim_weight    = self.weights.dim_weights[d_idx]
            dim_tfn       = self.weights.dim_fuzzy_weights[d_idx]
            crit_w_list   = self.weights.crit_weights[dim.code]
            crit_tfn_list = self.weights.crit_fuzzy_weights[dim.code]

            crit_results: List[CriterionResult] = []
            # Φ_D(xp) = Σ_c  w̄_c · s_c(xp)
            phi_d = 0.0

            for c_idx, crit in enumerate(dim.criteria):
                scorer    = CRITERION_SCORERS.get(crit.code, lambda _: 0.5)
                raw_score = scorer(player)
                crit_w    = crit_w_list[c_idx]
                contrib   = crit_w * raw_score
                phi_d    += contrib

                crit_results.append(CriterionResult(
                    code=crit.code, name=crit.name,
                    raw_score=raw_score, weight=crit_w,
                    weight_tfn=crit_tfn_list[c_idx].to_dict(),
                    contribution=contrib,
                ))

            # Contribution de cette dimension au score global
            dim_contrib = dim_weight * phi_d
            gamma      += dim_contrib

            dim_results.append(DimensionResult(
                code=dim.code, name=dim.name,
                weight=dim_weight, weight_tfn=dim_tfn.to_dict(),
                criteria=crit_results,
                dimension_score=phi_d,
                weighted_contribution=dim_contrib,
                cr=self.weights.crit_cr[dim.code],
            ))

        # Décision basée sur Γ(xp) — seuils standards
        if gamma >= 0.70:   decision = "expert"
        elif gamma >= 0.45: decision = "intermediate"
        else:               decision = "beginner"

        return EvaluationResult(
            player_id=player.get("player_id", -1),
            final_score=gamma,
            decision=decision,
            ml_level=player.get("level", "unknown"),   # label GMM, annotatif
            dimensions=dim_results,
            global_cr=self.weights.dim_cr,
        )


# ─────────────────────────────────────────────────────────────────────────────
# STATISTIQUES DE GROUPE ASF  (Eq. 2 du PDF)
# ─────────────────────────────────────────────────────────────────────────────

def compute_asf_group_stats(summaries: list) -> dict:
    """
    Calcule les statistiques agrégées par profil GMM.

    Γ*_L  = (1/|G_L|) Σ_{p∈G_L} Γ(xp)
    Φ̄_L_D = (1/|G_L|) Σ_{p∈G_L} Φ_D(xp)

    Returns:
        dict { 'beginner': {...}, 'intermediate': {...}, 'expert': {...} }
    """
    DIM_CODES = [dim.code for dim in DIMENSIONS]
    profiles  = {}

    for lv in ["beginner", "intermediate", "expert"]:
        grp = [s for s in summaries if s.get("ml_level","").lower() == lv]
        if not grp:
            profiles[lv] = {"count": 0}
            continue

        n = len(grp)
        avg_gamma = float(np.mean([s["fuzzy_score"] for s in grp]))

        dim_avgs = {}
        for d in DIM_CODES:
            vals = [s["dimension_scores"].get(d, 0.0) for s in grp]
            dim_avgs[f"avg_{d}"] = round(float(np.mean(vals)), 4)

        profiles[lv] = {
            "count"      : n,
            "avg_global" : round(avg_gamma, 4),
            **dim_avgs,
        }

    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_fuzzy_evaluation(db, weights: AHPWeights) -> int:
    """
    1. Charge les joueurs classifiés par le GMM (players_classified)
    2. Évalue chaque joueur via Fuzzy AHP ASF : Γ(xp) = Σ_D w̄_D · Φ_D(xp)
    3. Calcule les statistiques de groupe : Γ*_L et Φ̄_L_D
    4. Persiste dans MongoDB :
         • fuzzy_results       — résumé léger par joueur
         • fuzzy_evaluation    — détail complet par joueur
         • fuzzy_weights_cache — poids AHP en cache
    """
    players = list(db.players_classified.find({}, {"_id": 0}))
    if not players:
        print("⏳  No classified player data yet.")
        return 0

    evaluator = PlayerEvaluator(weights)
    summaries = []
    detailed  = []

    for player in players:
        result = evaluator.evaluate(player)
        rd     = result.to_dict()

        # Résumé léger pour fuzzy_results
        summaries.append({
            "player_id"          : result.player_id,
            "fuzzy_score"        : round(result.final_score, 4),
            "global_score"       : round(result.final_score, 4),
            "decision"           : result.decision,
            "ml_level"           : result.ml_level,       # label GMM annotatif
            "global_cr"          : round(result.global_cr, 4),
            "integration_method" : "ASF",
            # Scores de dimension Φ_D pour affichage rapide
            "dimension_scores"   : {
                d["code"]: round(d["dimension_score"], 4)
                for d in rd["dimensions"]
            },
            # Alias PD/TD/LD/BD pour compatibilité frontend existant
            **{d["code"]: round(d["dimension_score"], 4) for d in rd["dimensions"]},
            **{f'{d["code"]}_label': _dim_label(d["dimension_score"]) for d in rd["dimensions"]},
        })
        detailed.append(rd)

    # Statistiques de groupe ASF (Eq. 2)
    group_stats = compute_asf_group_stats(summaries)

    # Persister
    db.fuzzy_results.delete_many({})
    db.fuzzy_results.insert_many(summaries)

    db.fuzzy_evaluation.delete_many({})
    db.fuzzy_evaluation.insert_many(detailed)

    # Sauvegarder les stats de groupe séparément (accessible via API)
    db.asf_group_stats.delete_many({})
    db.asf_group_stats.insert_one({
        "groups"             : group_stats,
        "integration_method" : "ASF",
        "n_players"          : len(players),
        "description"        : (
            "Γ*_L = moyenne du score Fuzzy AHP par profil GMM. "
            "Le label GMM est annotatif et n'influe pas sur Γ(xp)."
        ),
    })

    # Cache des poids
    db.fuzzy_weights_cache.delete_many({})
    db.fuzzy_weights_cache.insert_one(weights.to_dict())

    print(f"  ✅  ASF evaluation done — {len(players)} players | groups: "
          f"{ {k: v.get('count',0) for k,v in group_stats.items()} }")
    return len(players)


def _dim_label(score: float) -> str:
    return ("Excellent" if score >= .80 else "Good" if score >= .65 else
            "Fair"      if score >= .50 else "Poor" if score >= .35 else "Very Poor")
