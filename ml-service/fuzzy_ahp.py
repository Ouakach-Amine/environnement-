from pymongo import MongoClient
import numpy as np
import time

client = MongoClient("mongodb://mongodb:27017/")
db = client["game_db"]

# ─────────────────────────────────────────────
# AHP Weights (must sum to 1)
# ─────────────────────────────────────────────
WEIGHTS = {
    "PD": 0.50,   # Pedagogical  – most important in a serious game
    "TD": 0.25,   # Technological
    "LD": 0.15,   # Ludic
    "BD": 0.10,   # Behavioural
}

# ─────────────────────────────────────────────
# Fuzzy membership functions  (return 0–1)
# ─────────────────────────────────────────────
def f_score(v):
    """High score → good learning outcome."""
    if v >= 80: return 1.0
    if v >= 60: return 0.8
    if v >= 40: return 0.5
    if v >= 20: return 0.3
    return 0.1

def f_success(v):
    return 1.0 if v == 1 else 0.1

def f_level(v):
    return min(v / 5.0, 1.0)

def f_response_time(rt):
    """Low response time → better technology."""
    if rt <= 0.5: return 1.0
    if rt <= 1.5: return 0.8
    if rt <= 3.0: return 0.5
    return 0.2

def f_errors(e):
    """Fewer errors → better."""
    if e == 0:  return 1.0
    if e <= 2:  return 0.8
    if e <= 5:  return 0.5
    return 0.2

def f_engagement(clicks, moves):
    """More interaction → more engagement (capped)."""
    return min((clicks + moves) / 100.0, 1.0)

def f_time(t):
    """Reasonable time spent → engaged but not stuck."""
    if 10 <= t <= 40: return 1.0
    if t < 10:        return 0.3   # too fast → not engaging
    if t <= 60:       return 0.7
    return 0.4                     # too long → frustrating

def f_repetition(r):
    """Moderate repetition shows practice; too much = frustration."""
    if r == 0:  return 0.5   # played once only
    if r <= 3:  return 1.0   # healthy practice
    if r <= 6:  return 0.6
    return 0.3               # stuck / bored


# ─────────────────────────────────────────────
# Dimension computations
# ─────────────────────────────────────────────
def compute_PD(p):
    """
    Pedagogical Dimension
    → Does the player learn? Improve? Succeed?
    """
    return (
        f_score(p.get("score", 0))   * 0.50 +
        f_success(p.get("success", 0)) * 0.30 +
        f_level(p.get("level", 1))   * 0.20
    )

def compute_TD(p):
    """
    Technological Dimension
    → Is the game technically responsive and error-free?
    """
    return (
        f_response_time(p.get("response_time", 0.0)) * 0.60 +
        f_errors(p.get("errors", 0))                  * 0.40
    )

def compute_LD(p):
    """
    Ludic Dimension
    → Is the game fun and engaging?
    """
    return (
        f_engagement(p.get("clicks", 0), p.get("moves", 0)) * 0.50 +
        f_time(p.get("time", 0))                              * 0.30 +
        f_repetition(p.get("repetition", 0))                 * 0.20
    )

def compute_BD(p):
    """
    Behavioural Dimension
    → Does the player show good learning behaviour?
    """
    return (
        f_errors(p.get("errors", 0))      * 0.50 +
        f_repetition(p.get("repetition", 0)) * 0.30 +
        f_success(p.get("success", 0))    * 0.20
    )


# ─────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────
def dim_label(s):
    if s >= 0.80: return "Excellent"
    if s >= 0.65: return "Good"
    if s >= 0.45: return "Fair"
    if s >= 0.25: return "Poor"
    return "Very Poor"

def player_decision(s):
    if s >= 0.75: return "expert"
    if s >= 0.50: return "intermediate"
    return "beginner"


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
raw_data = list(db.players.find({}, {"_id": 0}))

if len(raw_data) == 0:
    print("❌ No player data found")
    exit()

results = []

for p in raw_data:
    pd_score = round(compute_PD(p), 3)
    td_score = round(compute_TD(p), 3)
    ld_score = round(compute_LD(p), 3)
    bd_score = round(compute_BD(p), 3)

    final = round(
        pd_score * WEIGHTS["PD"] +
        td_score * WEIGHTS["TD"] +
        ld_score * WEIGHTS["LD"] +
        bd_score * WEIGHTS["BD"],
        3
    )

    results.append({
        "player_id"  : p["player_id"],
        # dimension scores
        "PD"         : pd_score,
        "TD"         : td_score,
        "LD"         : ld_score,
        "BD"         : bd_score,
        # labels per dimension
        "PD_label"   : dim_label(pd_score),
        "TD_label"   : dim_label(td_score),
        "LD_label"   : dim_label(ld_score),
        "BD_label"   : dim_label(bd_score),
        # overall
        "fuzzy_score": final,
        "ML_level"   : player_decision(final),   # filled below from ML results
    })

# ── enrich with ML cluster level ──
ml_map = {
    d["player_id"]: d.get("level", "—")
    for d in db.players_classified.find({}, {"_id": 0, "player_id": 1, "level": 1})
}
for r in results:
    r["ML_level"] = ml_map.get(r["player_id"], "—")


# ─────────────────────────────────────────────
# Game-level evaluation (aggregate over all players)
# ─────────────────────────────────────────────
def game_verdict(score):
    if score >= 0.80:
        return {
            "rating"     : "Excellent Serious Game ⭐⭐⭐⭐⭐",
            "satisfied"  : True,
            "learning"   : True,
            "summary"    : (
                "The game excels in all four dimensions. "
                "Players achieve strong learning outcomes, enjoy engaging gameplay, "
                "experience smooth technology, and demonstrate positive behaviour patterns. "
                "Players are highly satisfied and gain measurable knowledge/skills."
            )
        }
    if score >= 0.65:
        return {
            "rating"     : "Good Serious Game ⭐⭐⭐⭐",
            "satisfied"  : True,
            "learning"   : True,
            "summary"    : (
                "The game performs well across most dimensions. "
                "Players generally enjoy the experience and achieve learning goals. "
                "Minor improvements could further boost engagement or technical quality."
            )
        }
    if score >= 0.50:
        return {
            "rating"     : "Average Serious Game ⭐⭐⭐",
            "satisfied"  : True,
            "learning"   : False,
            "summary"    : (
                "The game delivers a reasonable experience but has notable gaps. "
                "Players may enjoy it but learning outcomes are inconsistent. "
                "Focus on strengthening the weakest dimensions."
            )
        }
    if score >= 0.35:
        return {
            "rating"     : "Below Average ⭐⭐",
            "satisfied"  : False,
            "learning"   : False,
            "summary"    : (
                "The game struggles to deliver on its serious-game promise. "
                "Players are not fully satisfied and learning is limited. "
                "Significant redesign is recommended."
            )
        }
    return {
        "rating"     : "Poor Serious Game ⭐",
        "satisfied"  : False,
        "learning"   : False,
        "summary"    : (
            "The game fails to meet serious-game standards. "
            "Players are frustrated and gain little. "
            "A comprehensive redesign of all four dimensions is needed."
        )
    }


avg_PD = float(np.mean([r["PD"] for r in results]))
avg_TD = float(np.mean([r["TD"] for r in results]))
avg_LD = float(np.mean([r["LD"] for r in results]))
avg_BD = float(np.mean([r["BD"] for r in results]))
avg_score = float(np.mean([r["fuzzy_score"] for r in results]))

verdict = game_verdict(avg_score)

# Per-dimension recommendations
def recommend(dim, score):
    recs = {
        "PD": {
            "Excellent": "Pedagogical design is outstanding — players achieve excellent learning outcomes.",
            "Good"     : "Good learning outcomes. Consider adding adaptive difficulty for further gains.",
            "Fair"     : "Learning outcomes are moderate. Strengthen feedback loops and level progression.",
            "Poor"     : "Players are not learning effectively. Revise pedagogical objectives and content.",
            "Very Poor": "Urgent redesign of learning content and objectives required.",
        },
        "TD": {
            "Excellent": "Technology is fast, stable and transparent to the player.",
            "Good"     : "Good technical quality. Minor optimisation of response times would help.",
            "Fair"     : "Technical issues are noticeable. Reduce errors and improve response time.",
            "Poor"     : "Frequent errors or lag degrade the experience. Significant tech fixes needed.",
            "Very Poor": "Critical technical failures. The game is difficult to play as-is.",
        },
        "LD": {
            "Excellent": "Players are highly engaged — the game is genuinely fun.",
            "Good"     : "Good engagement. More interactive elements could deepen immersion.",
            "Fair"     : "Moderate fun factor. Add more varied mechanics or narrative elements.",
            "Poor"     : "Players find the game boring or repetitive. Game design overhaul advised.",
            "Very Poor": "Players disengage quickly. Core game loop needs rethinking.",
        },
        "BD": {
            "Excellent": "Players show excellent learning behaviour — persistent, low-error, successful.",
            "Good"     : "Good behavioural patterns. Encourage strategic thinking to reach excellence.",
            "Fair"     : "Mixed behaviours. Introduce scaffolding and in-game hints to guide players.",
            "Poor"     : "Players show frustration or disengagement. Add support mechanisms.",
            "Very Poor": "Players are completely lost or disengaged. Foundational UX review required.",
        },
    }
    label = dim_label(score)
    return recs[dim].get(label, "")

game_eval = {
    "avg_PD"          : round(avg_PD, 3),
    "avg_TD"          : round(avg_TD, 3),
    "avg_LD"          : round(avg_LD, 3),
    "avg_BD"          : round(avg_BD, 3),
    "avg_score"       : round(avg_score, 3),
    "PD_label"        : dim_label(avg_PD),
    "TD_label"        : dim_label(avg_TD),
    "LD_label"        : dim_label(avg_LD),
    "BD_label"        : dim_label(avg_BD),
    "PD_recommendation": recommend("PD", avg_PD),
    "TD_recommendation": recommend("TD", avg_TD),
    "LD_recommendation": recommend("LD", avg_LD),
    "BD_recommendation": recommend("BD", avg_BD),
    "rating"          : verdict["rating"],
    "player_satisfied": verdict["satisfied"],
    "learning_achieved": verdict["learning"],
    "summary"         : verdict["summary"],
    "n_players"       : len(results),
    "timestamp"       : time.time(),
}

# Save
db.fuzzy_results.delete_many({})
db.fuzzy_results.insert_many(results)

db.game_evaluation.delete_many({})
db.game_evaluation.insert_one(game_eval)

print(f"✅ Fuzzy AHP done — {len(results)} players evaluated")
print(f"📊 Game rating : {game_eval['rating']}")
print(f"   PD={avg_PD:.2f}  TD={avg_TD:.2f}  LD={avg_LD:.2f}  BD={avg_BD:.2f}")
print(f"   Overall={avg_score:.2f}")
