"""
fuzzy_models.py
───────────────────────────────────────────────────────────────────────────────
Data structures for the Fuzzy AHP Serious Game Evaluation System.

Reference: "Serious Game Evaluation System based on Fuzzy AHP"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time


# ─────────────────────────────────────────────────────────────────────────────
# TRIANGULAR FUZZY NUMBER (TFN)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TFN:
    """
    Triangular Fuzzy Number represented as (l, m, u) where:
        l  = lower bound (pessimistic value)
        m  = modal value (most likely)
        u  = upper bound (optimistic value)
    Constraint: l ≤ m ≤ u
    """
    l: float
    m: float
    u: float

    def __post_init__(self):
        if not (self.l <= self.m <= self.u):
            raise ValueError(f"Invalid TFN: l={self.l}, m={self.m}, u={self.u} — must satisfy l ≤ m ≤ u")

    # ── Arithmetic operations ────────────────────────────────────────────────

    def reciprocal(self) -> "TFN":
        """
        Reciprocal: ã⁻¹ = (1/u, 1/m, 1/l)
        Used to fill the lower-triangle of the comparison matrix.
        """
        return TFN(1.0 / self.u, 1.0 / self.m, 1.0 / self.l)

    def __mul__(self, other: "TFN") -> "TFN":
        """Fuzzy multiplication: (l₁·l₂, m₁·m₂, u₁·u₂)"""
        return TFN(self.l * other.l, self.m * other.m, self.u * other.u)

    def __add__(self, other: "TFN") -> "TFN":
        """Fuzzy addition: (l₁+l₂, m₁+m₂, u₁+u₂)"""
        return TFN(self.l + other.l, self.m + other.m, self.u + other.u)

    # ── Defuzzification ──────────────────────────────────────────────────────

    def defuzzify(self) -> float:
        """
        Centroid defuzzification: M = (l + m + u) / 3
        Converts the fuzzy weight into a crisp scalar.
        """
        return (self.l + self.m + self.u) / 3.0

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"l": round(self.l, 6), "m": round(self.m, 6), "u": round(self.u, 6)}

    def __repr__(self) -> str:
        return f"TFN({self.l:.4f}, {self.m:.4f}, {self.u:.4f})"


# ─────────────────────────────────────────────────────────────────────────────
# LINGUISTIC SCALE → TFN MAPPING  (Saaty's 9-point scale, adapted for fuzzy)
# ─────────────────────────────────────────────────────────────────────────────

#  Term          | Crisp value | TFN (l, m, u)
# ───────────────|─────────────|───────────────
#  equal         |      1      | (1, 1, 1)
#  weak          |      2      | (1, 2, 3)
#  moderate      |      3      | (2, 3, 4)
#  strong        |      4      | (3, 4, 5)
#  high          |      5      | (4, 5, 6)
#  very_high     |      7      | (6, 7, 8)
#  extreme       |      9      | (8, 9, 9)

LINGUISTIC_SCALE: Dict[str, Tuple[float, float, float]] = {
    "equal":     (1, 1, 1),
    "weak":      (1, 2, 3),
    "moderate":  (2, 3, 4),
    "strong":    (3, 4, 5),
    "high":      (4, 5, 6),
    "very_high": (6, 7, 8),
    "extreme":   (8, 9, 9),
}


def make_tfn(l: float, m: float, u: float) -> TFN:
    return TFN(l, m, u)


def linguistic_to_tfn(term: str) -> TFN:
    """Convert a linguistic importance term to its TFN."""
    if term not in LINGUISTIC_SCALE:
        raise ValueError(f"Unknown linguistic term '{term}'. Valid: {list(LINGUISTIC_SCALE)}")
    l, m, u = LINGUISTIC_SCALE[term]
    return TFN(l, m, u)


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONS AND CRITERIA
# Criteria within each dimension are ordered by DECREASING importance.
# This ordering drives the pairwise comparison matrices defined in
# fuzzy_evaluation.py (all upper-triangle values ≥ 1 by convention).
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Criterion:
    code: str
    name: str
    description: str = ""


@dataclass
class Dimension:
    code: str
    name: str
    criteria: List[Criterion]
    description: str = ""


DIMENSIONS: List[Dimension] = [

    # ── PD – Pedagogical ─────────────────────────────────────────────────────
    # Importance order: Lr > Ts > Pc > Em
    Dimension(
        code="PD",
        name="Pedagogical",
        description="Evaluates educational effectiveness and learning outcomes",
        criteria=[
            Criterion("Lr", "Learning Result",           "Actual learning achieved by the player"),
            Criterion("Ts", "Targeted Skills",           "Skills the game aims to develop"),
            Criterion("Pc", "Pedagogical Consideration", "Quality of the pedagogical design"),
            Criterion("Em", "Error Management",          "How errors are handled and used for learning"),
        ],
    ),

    # ── TD – Technological ───────────────────────────────────────────────────
    # Importance order: Ui > P > U > Gd
    Dimension(
        code="TD",
        name="Technological",
        description="Evaluates technical quality and user interface design",
        criteria=[
            Criterion("Ui", "User Interface", "Clarity and intuitiveness of the UI"),
            Criterion("P",  "Performance",    "Technical performance and responsiveness"),
            Criterion("U",  "Usability",      "Ease of use and accessibility"),
            Criterion("Gd", "Game Design",    "Visual and interaction design quality"),
        ],
    ),

    # ── LD – Ludic ───────────────────────────────────────────────────────────
    # Importance order: C > G > F > I
    Dimension(
        code="LD",
        name="Ludic",
        description="Evaluates enjoyment, challenge, and immersion factors",
        criteria=[
            Criterion("C", "Challenge",  "Appropriate difficulty and challenge level"),
            Criterion("G", "Gameplay",   "Quality of game mechanics"),
            Criterion("F", "Fun",        "Entertainment and enjoyment value"),
            Criterion("I", "Immersion",  "Depth of player immersion"),
        ],
    ),

    # ── BD – Behavioural ─────────────────────────────────────────────────────
    # Importance order: M > E > Ue
    Dimension(
        code="BD",
        name="Behavioural",
        description="Evaluates player behaviour and psychological engagement",
        criteria=[
            Criterion("M",  "Motivation",      "Player motivation to continue playing"),
            Criterion("E",  "Engagement",      "Depth of player engagement"),
            Criterion("Ue", "User Experience", "Overall user experience quality"),
        ],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# RESULT STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CriterionResult:
    code: str
    name: str
    raw_score: float          # Normalized player score for this criterion [0, 1]
    weight: float             # Fuzzy AHP weight for this criterion
    weight_tfn: dict          # TFN dict {l, m, u}
    contribution: float       # weight × raw_score


@dataclass
class DimensionResult:
    code: str
    name: str
    weight: float             # Fuzzy AHP weight for this dimension
    weight_tfn: dict          # TFN dict {l, m, u}
    criteria: List[CriterionResult]
    dimension_score: float    # Σ (criterion_weight × criterion_score)
    weighted_contribution: float  # dimension_weight × dimension_score
    cr: float                 # Consistency Ratio for this sub-matrix


@dataclass
class EvaluationResult:
    player_id: int
    final_score: float        # Global weighted score [0, 1]
    decision: str             # "expert" | "intermediate" | "beginner"
    ml_level: str             # Level from ML clustering (for comparison)
    dimensions: List[DimensionResult]
    global_cr: float          # CR of the dimension-level matrix
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "player_id":  self.player_id,
            "final_score": round(self.final_score, 4),
            "decision":   self.decision,
            "ml_level":   self.ml_level,
            "global_cr":  round(self.global_cr, 4),
            "dimensions": [
                {
                    "code":  d.code,
                    "name":  d.name,
                    "weight": round(d.weight, 4),
                    "weight_tfn": d.weight_tfn,
                    "dimension_score": round(d.dimension_score, 4),
                    "weighted_contribution": round(d.weighted_contribution, 4),
                    "cr": round(d.cr, 4),
                    "criteria": [
                        {
                            "code":  c.code,
                            "name":  c.name,
                            "raw_score":    round(c.raw_score, 4),
                            "weight":       round(c.weight, 4),
                            "weight_tfn":   c.weight_tfn,
                            "contribution": round(c.contribution, 4),
                        }
                        for c in d.criteria
                    ],
                }
                for d in self.dimensions
            ],
            "timestamp": self.timestamp,
        }
