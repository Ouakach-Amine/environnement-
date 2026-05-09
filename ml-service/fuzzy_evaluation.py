"""
fuzzy_evaluation.py
───────────────────────────────────────────────────────────────────────────────
Serious Game Evaluation using Fuzzy AHP.

This module:
  1.  Defines the expert pairwise comparison matrices (dimensions + criteria)
  2.  Computes Fuzzy AHP weights (once at startup)
  3.  Maps raw player game-data to normalized criterion scores [0, 1]
  4.  Combines weights × scores to produce a final evaluation per player
  5.  Reads from MongoDB: players_classified, model_metrics
  6.  Writes to  MongoDB: fuzzy_results, fuzzy_evaluation, fuzzy_weights_cache
"""

import math
from typing import Dict, List, Optional, Tuple
from pymongo import MongoClient

from fuzzy_models import (
    TFN, DIMENSIONS, Dimension, Criterion,
    CriterionResult, DimensionResult, EvaluationResult,
)
from fuzzy_ahp import FuzzyAHP


# ─────────────────────────────────────────────────────────────────────────────
# HELPER : TFN FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def _t(l, m, u) -> TFN:
    return TFN(float(l), float(m), float(u))


def _r(l, m, u) -> TFN:
    """Reciprocal TFN — used when column criterion outranks row criterion."""
    return TFN(1.0 / u, 1.0 / m, 1.0 / l)


# ─────────────────────────────────────────────────────────────────────────────
# PAIRWISE COMPARISON MATRICES
#
# Convention:
#   • Criteria are listed in DECREASING importance order (see fuzzy_models.py).
#   • The upper triangle therefore always has values ≥ 1 (row more important
#     than column), so every TFN (l, m, u) satisfies l ≥ 1.
#   • Values are based on the Saaty 9-point linguistic scale mapped to TFNs:
#       equal=(1,1,1)  weak=(1,2,3)  moderate=(2,3,4)  strong=(3,4,5)
#       high=(4,5,6)   very_high=(6,7,8)  extreme=(8,9,9)
#
# DIMENSION LEVEL  (PD, TD, LD, BD)
#   Importance rationale: Pedagogical learning is the primary purpose
#   of a Serious Game, followed by Technology, Ludic design, and Behaviour.
#
# CRITERIA LEVEL   (per dimension)
#   Importance rationale derived from the paper's pedagogical focus.
# ─────────────────────────────────────────────────────────────────────────────

# Shorthand aliases
EQ   = _t(1, 1, 1)
WK   = _t(1, 2, 3)
MOD  = _t(2, 3, 4)
STR  = _t(3, 4, 5)
HI   = _t(4, 5, 6)
VH   = _t(6, 7, 8)
EXT  = _t(8, 9, 9)

# ── Dimension-level comparisons (4 criteria → 6 upper-triangle values) ───────
#
#           PD     TD     LD     BD
#   PD  [  1.0    MOD    HI     VH  ]
#   TD  [ rMOD    1.0    MOD    STR ]
#   LD  [ rHI    rMOD    1.0    WK  ]
#   BD  [ rVH    rSTR   rWK     1.0 ]
#
DIMENSION_UPPER: List[TFN] = [
    MOD,   # PD vs TD
    HI,    # PD vs LD
    VH,    # PD vs BD
    MOD,   # TD vs LD
    STR,   # TD vs BD
    WK,    # LD vs BD
]

# ── PD criteria: [Lr, Ts, Pc, Em] ────────────────────────────────────────────
#           Lr     Ts     Pc     Em
#   Lr  [  1.0    WK     MOD    STR ]
#   Ts  [ rWK     1.0    WK     MOD ]
#   Pc  [ rMOD   rWK     1.0    WK  ]
#   Em  [ rSTR   rMOD   rWK     1.0 ]
#
PD_UPPER: List[TFN] = [
    WK,    # Lr vs Ts
    MOD,   # Lr vs Pc
    STR,   # Lr vs Em
    WK,    # Ts vs Pc
    MOD,   # Ts vs Em
    WK,    # Pc vs Em
]

# ── TD criteria: [Ui, P, U, Gd] ──────────────────────────────────────────────
TD_UPPER: List[TFN] = [
    WK,    # Ui vs P
    MOD,   # Ui vs U
    MOD,   # Ui vs Gd
    WK,    # P  vs U
    WK,    # P  vs Gd
    WK,    # U  vs Gd
]

# ── LD criteria: [C, G, F, I] ────────────────────────────────────────────────
LD_UPPER: List[TFN] = [
    WK,    # C vs G
    MOD,   # C vs F
    MOD,   # C vs I
    WK,    # G vs F
    WK,    # G vs I
    WK,    # F vs I
]

# ── BD criteria: [M, E, Ue] — 3×3 → 3 upper-triangle values ─────────────────
BD_UPPER: List[TFN] = [
    WK,    # M vs E
    MOD,   # M vs Ue
    WK,    # E vs Ue
]

# Mapping: dimension code → upper triangle TFNs
CRITERIA_UPPER_TRIANGLES: Dict[str, List[TFN]] = {
    "PD": PD_UPPER,
    "TD": TD_UPPER,
    "LD": LD_UPPER,
    "BD": BD_UPPER,
}


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT COMPUTATION  (runs once at startup)
# ─────────────────────────────────────────────────────────────────────────────

class AHPWeights:
    """
    Computes and stores Fuzzy AHP weights for all dimensions and criteria.
    Instantiate once; weights are deterministic (fixed comparison matrices).
    """

    def __init__(self):
        self.ahp = FuzzyAHP()

        print("\n═══════════════════════════════════════════")
        print("  Computing Fuzzy AHP Weights")
        print("═══════════════════════════════════════════")

        # ── Dimension weights ──────────────────────────────────────────────
        print("\n▸ Dimension-level matrix (PD, TD, LD, BD)")
        dim_result = self.ahp.run(DIMENSION_UPPER, n=4)
        self.dim_cr:             float       = dim_result["cr"]
        self.dim_fuzzy_weights:  List[TFN]   = dim_result["fuzzy_weights"]
        self.dim_weights:        List[float] = dim_result["normalized_weights"]

        # ── Criteria weights (per dimension) ──────────────────────────────
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
        print(f"  Dimension weights:")
        for dim, w in zip(DIMENSIONS, self.dim_weights):
            print(f"    {dim.code}: {w:.4f}")
        print("═══════════════════════════════════════════\n")

    def to_dict(self) -> dict:
        """Serialize all weights for MongoDB / API response."""
        out = {
            "dimension_cr": round(self.dim_cr, 4),
            "dimensions": [],
        }
        for i, dim in enumerate(DIMENSIONS):
            d_entry = {
                "code":   dim.code,
                "name":   dim.name,
                "weight": round(self.dim_weights[i], 4),
                "weight_tfn": self.dim_fuzzy_weights[i].to_dict(),
                "cr":     round(self.crit_cr[dim.code], 4),
                "criteria": [],
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
# CRITERION SCORE FUNCTIONS
# Each function receives the player document and returns a score in [0, 1].
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _ml_level_score(level_str: str) -> float:
    """Convert ML-classified level string → numeric score."""
    return {"beginner": 0.25, "intermediate": 0.60, "expert": 1.00}.get(
        str(level_str).lower(), 0.5
    )


# ── PD  [Lr, Ts, Pc, Em] ─────────────────────────────────────────────────────

def score_Lr(p: dict) -> float:
    """Learning Result: directly mapped to normalized game score."""
    return _clamp(p.get("score", 0) / 100.0)


def score_Ts(p: dict) -> float:
    """Targeted Skills: higher game level → more skills engaged."""
    return _clamp(p.get("level", 1) / 10.0)


def score_Pc(p: dict) -> float:
    """Pedagogical Consideration: success rate (0–1 or 0/1)."""
    raw = p.get("success", 0)
    return _clamp(float(raw) if raw <= 1 else raw / 100.0)


def score_Em(p: dict) -> float:
    """Error Management: fewer errors → better error-management quality."""
    errors = p.get("errors", 0)
    return _clamp(1.0 - errors / 10.0)


# ── TD  [Ui, P, U, Gd] ───────────────────────────────────────────────────────

def score_Ui(p: dict) -> float:
    """User Interface: moderate click count signals intuitive UI."""
    clicks = p.get("clicks", 0)
    # Bell-curve: peak at 50 clicks; decays beyond 150
    ideal = 50.0
    spread = 60.0
    return _clamp(math.exp(-0.5 * ((clicks - ideal) / spread) ** 2))


def score_P(p: dict) -> float:
    """Performance: low response time → high performance."""
    rt = p.get("response_time", 1.0)
    return _clamp(1.0 - rt / 5.0)


def score_U(p: dict) -> float:
    """Usability: faster completion (time) → better usability."""
    t = p.get("time", 60)
    return _clamp(1.0 - t / 120.0)


def score_Gd(p: dict) -> float:
    """Game Design: rich exploration (moves) → good game design engagement."""
    moves = p.get("moves", 0)
    return _clamp(moves / 50.0)


# ── LD  [C, G, F, I] ─────────────────────────────────────────────────────────

def score_C(p: dict) -> float:
    """
    Challenge: appropriate when the player succeeds despite some errors.
    High score + moderate errors = just-right challenge.
    """
    s = p.get("score", 0) / 100.0
    e = _clamp(p.get("errors", 0) / 10.0)
    # Maximize when score is high AND errors are moderate (not zero, not too many)
    challenge = s * (1 - abs(e - 0.3) / 0.7)
    return _clamp(challenge)


def score_G(p: dict) -> float:
    """Gameplay: fluent play measured by moves-per-click efficiency."""
    clicks = max(p.get("clicks", 1), 1)
    moves  = p.get("moves", 0)
    ratio  = moves / clicks
    return _clamp(ratio / 2.0)  # ideal ratio ≈ 2 moves per click


def score_F(p: dict) -> float:
    """Fun: repeated play sessions indicate the game is enjoyable."""
    return _clamp(p.get("repetition", 0) / 5.0)


def score_I(p: dict) -> float:
    """Immersion: blend of session time and repetition."""
    time_score = _clamp(p.get("time", 0) / 60.0)
    rep_score  = _clamp(p.get("repetition", 0) / 5.0)
    return (time_score + rep_score) / 2.0


# ── BD  [M, E, Ue] ───────────────────────────────────────────────────────────

def score_M(p: dict) -> float:
    """Motivation: voluntary repetition is the strongest motivation signal."""
    return _clamp(p.get("repetition", 0) / 5.0)


def score_E(p: dict) -> float:
    """Engagement: total interaction depth (clicks + moves)."""
    return _clamp((p.get("clicks", 0) + p.get("moves", 0)) / 100.0)


def score_Ue(p: dict) -> float:
    """User Experience: composite of score quality and error cleanliness."""
    s = p.get("score", 0) / 100.0
    e = _clamp(1.0 - p.get("errors", 0) / 10.0)
    return (s + e) / 2.0


# ── Score dispatcher ─────────────────────────────────────────────────────────

CRITERION_SCORERS = {
    "Lr": score_Lr, "Ts": score_Ts, "Pc": score_Pc, "Em": score_Em,
    "Ui": score_Ui, "P":  score_P,  "U":  score_U,  "Gd": score_Gd,
    "C":  score_C,  "G":  score_G,  "F":  score_F,  "I":  score_I,
    "M":  score_M,  "E":  score_E,  "Ue": score_Ue,
}


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class PlayerEvaluator:
    """
    Evaluates each player document using pre-computed Fuzzy AHP weights.

    Input  : player dict from MongoDB players_classified collection
    Output : EvaluationResult with detailed per-dimension/criterion breakdown
    """

    def __init__(self, weights: AHPWeights):
        self.weights = weights

    def evaluate(self, player: dict) -> EvaluationResult:
        """Full Fuzzy AHP evaluation for a single player."""
        dim_results: List[DimensionResult] = []
        final_score = 0.0

        for d_idx, dim in enumerate(DIMENSIONS):
            dim_weight     = self.weights.dim_weights[d_idx]
            dim_tfn        = self.weights.dim_fuzzy_weights[d_idx]
            crit_w_list    = self.weights.crit_weights[dim.code]
            crit_tfn_list  = self.weights.crit_fuzzy_weights[dim.code]

            # Compute per-criterion scores
            crit_results: List[CriterionResult] = []
            dim_score = 0.0

            for c_idx, crit in enumerate(dim.criteria):
                scorer    = CRITERION_SCORERS.get(crit.code, lambda _: 0.5)
                raw_score = scorer(player)
                crit_w    = crit_w_list[c_idx]
                contrib   = crit_w * raw_score
                dim_score += contrib

                crit_results.append(CriterionResult(
                    code=crit.code,
                    name=crit.name,
                    raw_score=raw_score,
                    weight=crit_w,
                    weight_tfn=crit_tfn_list[c_idx].to_dict(),
                    contribution=contrib,
                ))

            weighted_contrib = dim_weight * dim_score
            final_score     += weighted_contrib

            dim_results.append(DimensionResult(
                code=dim.code,
                name=dim.name,
                weight=dim_weight,
                weight_tfn=dim_tfn.to_dict(),
                criteria=crit_results,
                dimension_score=dim_score,
                weighted_contribution=weighted_contrib,
                cr=self.weights.crit_cr[dim.code],
            ))

        # Decision thresholds
        if final_score >= 0.70:
            decision = "expert"
        elif final_score >= 0.45:
            decision = "intermediate"
        else:
            decision = "beginner"

        return EvaluationResult(
            player_id=player.get("player_id", -1),
            final_score=final_score,
            decision=decision,
            ml_level=player.get("level", "unknown"),
            dimensions=dim_results,
            global_cr=self.weights.dim_cr,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MONGODB I/O
# ─────────────────────────────────────────────────────────────────────────────

def run_fuzzy_evaluation(db, weights: AHPWeights) -> int:
    """
    Load players_classified from MongoDB, evaluate each player,
    and persist results to:
        • fuzzy_results       — lightweight summary (player_id, score, decision)
        • fuzzy_evaluation    — full detailed breakdown
        • fuzzy_weights_cache — AHP weights snapshot for the frontend
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
        result_dict = result.to_dict()

        # Lightweight summary for the original fuzzy_results collection
        summaries.append({
            "player_id":   result.player_id,
            "fuzzy_score": round(result.final_score, 4),
            "decision":    result.decision,
            "ml_level":    result.ml_level,
            "global_cr":   round(result.global_cr, 4),
            # Per-dimension scores for quick display
            "dimension_scores": {
                d["code"]: round(d["dimension_score"], 4)
                for d in result_dict["dimensions"]
            },
        })
        detailed.append(result_dict)

    # Persist
    db.fuzzy_results.delete_many({})
    db.fuzzy_results.insert_many(summaries)

    db.fuzzy_evaluation.delete_many({})
    db.fuzzy_evaluation.insert_many(detailed)

    # Cache weights
    db.fuzzy_weights_cache.delete_many({})
    db.fuzzy_weights_cache.insert_one(weights.to_dict())

    print(f"  ✅  Fuzzy evaluation done — {len(players)} players processed")
    return len(players)
