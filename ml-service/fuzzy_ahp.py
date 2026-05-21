"""
Fuzzy AHP — Évaluation du Serious Game
Méthode : Average Score Fusion (ASF)

Le niveau ML (GMM) est attaché comme label descriptif uniquement.
Il n'influence PAS le calcul du score numérique (pas de PPIF).
Score final : Γ(xp) = Σ_D  w̄_D · Φ_D(xp)

Les statistiques agrégées par groupe (Beginner / Intermediate / Expert)
sont calculées a posteriori pour l'analyse et la visualisation.

Lit   : db.players + db.players_classified
Écrit : db.fuzzy_results + db.game_evaluation
"""
import time, math
import numpy as np
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db     = client["game_db"]

# ════════════════════════════════════════════════════════════════════
# TFN ALGEBRA
# ════════════════════════════════════════════════════════════════════
def tfn_add(ts):  return (sum(t[0] for t in ts), sum(t[1] for t in ts), sum(t[2] for t in ts))
def tfn_mul(a,b): return (a[0]*b[0], a[1]*b[1], a[2]*b[2])
def tfn_inv(a):   return (1/a[2], 1/a[1], 1/a[0])
def tfn_geo(row):
    n=len(row)
    return (math.prod(t[0] for t in row)**(1/n),
            math.prod(t[1] for t in row)**(1/n),
            math.prod(t[2] for t in row)**(1/n))
def defuzz(t): return (t[0]+t[1]+t[2])/3.0

_SC = {1:(1,1,1),2:(1,2,3),3:(2,3,4),4:(3,4,5),5:(4,5,6),
       6:(5,6,7),7:(6,7,8),8:(7,8,9),9:(8,9,9)}
def sc(v): return _SC[int(round(v))] if v>=1 else tfn_inv(_SC[int(round(1/v))])

RI_TABLE = {1:0,2:0,3:.58,4:.90,5:1.12,6:1.24,7:1.32,8:1.41}

def run_ahp(M, labels):
    n  = len(M)
    Mf = [[sc(M[i][j]) for j in range(n)] for i in range(n)]
    Mm = np.array([[Mf[i][j][1] for j in range(n)] for i in range(n)], dtype=float)
    w  = (Mm / Mm.sum(axis=0)).mean(axis=1)
    lm = float(np.mean((Mm @ w) / w))
    ci = (lm - n) / (n - 1) if n > 1 else 0
    ri = RI_TABLE.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0
    gm = [tfn_geo(Mf[i]) for i in range(n)]
    ti = tfn_inv(tfn_add(gm))
    fw = [tfn_mul(g, ti) for g in gm]
    cw = [defuzz(f) for f in fw]
    s  = sum(cw)
    nw = [c / s for c in cw]
    td = lambda t: {"l":round(t[0],4),"m":round(t[1],4),"u":round(t[2],4)}
    return {
        "labels"       : labels,
        "lambda_max"   : round(lm, 4),
        "CI"           : round(ci, 4),
        "CR"           : round(cr, 4),
        "consistent"   : bool(cr <= 0.10),
        "geo_means"    : [td(g) for g in gm],
        "fuzzy_weights": [td(f) for f in fw],
        "norm_weights" : [round(w, 4) for w in nw],
        "weight_map"   : {labels[i]: round(nw[i], 4) for i in range(n)},
    }

# ════════════════════════════════════════════════════════════════════
# MATRICES DE COMPARAISON (inchangées)
# ════════════════════════════════════════════════════════════════════
DIM_M = [[1,3,5,7],[1/3,1,3,5],[1/5,1/3,1,3],[1/7,1/5,1/3,1]]
PD_M  = [[1,2,1/2,3],[1/2,1,1/3,2],[2,3,1,5],[1/3,1/2,1/5,1]]
TD_M  = [[1,1/2,2,1/2],[2,1,3,2],[1/2,1/3,1,1/2],[2,1/2,2,1]]
LD_M  = [[1,1,2,3],[1,1,2,3],[1/2,1/2,1,2],[1/3,1/3,1/2,1]]
BD_M  = [[1,2,3],[1/2,1,2],[1/3,1/2,1]]

DIM_L = ["PD","TD","LD","BD"]
PD_L  = ["Ts","Pc","Lr","Em"]
TD_L  = ["Gd","P","Ui","U"]
LD_L  = ["C","F","G","I"]
BD_L  = ["M","E","Ue"]

# ════════════════════════════════════════════════════════════════════
# FONCTIONS D'APPARTENANCE (inchangées)
# ════════════════════════════════════════════════════════════════════
c = lambda v: max(0., min(1., float(v)))

def f_Ts(p):
    lv  = c(p.get("level", 1) / 5.0)
    obj = c(p.get("objectives_completed", 0) / 5.0)
    prog= c(p.get("progression_rate", 0.0))
    return c(lv * 0.40 + obj * 0.35 + prog * 0.25)

def f_Pc(p):
    ratio = c(p.get("score", 0) / max(p.get("level", 1) * 20.0, 1))
    ks    = c(p.get("knowledge_score", 0.0) / 100.0)
    return c(ratio * 0.50 + ks * 0.50)

def f_Lr(p):
    sc  = c(p.get("score", 0) / 100.0)
    ok  = c(p.get("correct_answers", 0) /
             max(p.get("correct_answers", 0) + p.get("wrong_answers", 1), 1))
    suc = float(p.get("success", 0))
    return c(sc * 0.40 + ok * 0.35 + suc * 0.25)

def f_Em(p):
    e   = p.get("errors", 0)
    wa  = p.get("wrong_answers", 0)
    ret = p.get("retry_after_fail", 0)
    err_score = (1.0 if e==0 else .85 if e<=2 else .60 if e<=5 else .35 if e<=9 else .15)
    wa_score  = c(1.0 - wa / max(wa + p.get("correct_answers", 1), 1))
    ret_score = (1.0 if ret==0 else .80 if ret<=2 else .60 if ret<=5 else .30)
    return c(err_score * 0.40 + wa_score * 0.35 + ret_score * 0.25)

def f_Gd(p):
    return c((p.get("clicks", 0) + p.get("moves", 0) + p.get("combo_count", 0)) / 100.0)

def f_P(p):
    rt  = p.get("response_time", 0.0)
    rt_s= (1.0 if rt<=.3 else .9 if rt<=.8 else .75 if rt<=1.5 else .5 if rt<=3 else .2)
    lag = p.get("lag_events", 0)
    lag_s = (1.0 if lag==0 else .85 if lag<=2 else .60 if lag<=5 else .30)
    fd  = p.get("frame_drops", 0)
    fd_s  = (1.0 if fd==0 else .85 if fd<=3 else .55 if fd<=8 else .25)
    cr  = p.get("crash_count", 0)
    cr_s  = (1.0 if cr==0 else .50 if cr==1 else .10)
    return c(rt_s * 0.35 + lag_s * 0.25 + fd_s * 0.25 + cr_s * 0.15)

def f_Ui(p):
    tot = p.get("clicks", 0) + p.get("moves", 0)
    bal = (0.5 if tot == 0 else
           c(1 - abs(p.get("clicks", 0) / tot - 0.5) * 1.6))
    ae  = p.get("api_errors", 0)
    ae_s= (1.0 if ae==0 else .80 if ae<=1 else .50 if ae<=3 else .20)
    return c(bal * 0.70 + ae_s * 0.30)

def f_U(p):
    lt  = p.get("load_time", 0.0)
    lt_s= (1.0 if lt<=1 else .85 if lt<=2 else .65 if lt<=4 else .35 if lt<=7 else .15)
    return c(lt_s * 0.50 + f_Em(p) * 0.50)

def f_C(p):
    lv  = c(p.get("level", 1) / 5.0)
    sc  = c(p.get("score", 0) / 100.0)
    ch  = c(p.get("challenges_attempted", 0) / 5.0)
    return c(lv * 0.40 + sc * 0.35 + ch * 0.25)

def f_F(p):
    r = p.get("repetition", 0)
    rep_s = (0.30 if r==0 else .70 if r<=2 else 1.0 if r<=4 else .75 if r<=7 else .45)
    bon = c(p.get("bonus_collected", 0) / 5.0)
    pvol= c(p.get("playtime_voluntary", 0) / 60.0)
    return c(rep_s * 0.50 + bon * 0.25 + pvol * 0.25)

def f_G(p):
    inter = c((p.get("moves", 0) + p.get("clicks", 0)) / 100.0)
    expl  = c(p.get("exploration_rate", 0.0))
    return c(inter * 0.60 + expl * 0.40)

def f_I(p):
    t = p.get("time", 0)
    t_s = (1.0 if 15<=t<=45 else .70 if (10<=t<15 or 45<t<=70) else .30 if t<10 else .40)
    sk  = p.get("skip_count", 0)
    sk_s= (1.0 if sk==0 else .80 if sk<=1 else .50 if sk<=3 else .20)
    idle= p.get("idle_time", 0.0)
    id_s= (1.0 if idle<=5 else .80 if idle<=15 else .50 if idle<=30 else .20)
    return c(t_s * 0.40 + sk_s * 0.30 + id_s * 0.30)

def f_M(p):
    rep = c(p.get("repetition", 0) / 5.0)
    da  = c(p.get("days_active", 1) / 7.0)
    ses = c(p.get("session_count", 1) / 5.0)
    return c(rep * 0.40 + da * 0.35 + ses * 0.25)

def f_E(p):
    ft  = c(p.get("focus_time", 0.0) / 60.0)
    inter = c((p.get("clicks", 0) + p.get("moves", 0)) / 80.0)
    pvol= c(p.get("playtime_voluntary", 0) / 60.0)
    return c(ft * 0.45 + inter * 0.35 + pvol * 0.20)

def f_Ue(p):
    fr  = p.get("frustration_events", 0)
    fr_s= (1.0 if fr==0 else .80 if fr<=2 else .50 if fr<=5 else .20)
    gu  = p.get("give_up_count", 0)
    gu_s= (1.0 if gu==0 else .70 if gu==1 else .40 if gu<=3 else .10)
    pc  = p.get("pause_count", 0)
    pc_s= (1.0 if pc==0 else .90 if pc<=2 else .70 if pc<=5 else .50)
    return c(fr_s * 0.45 + gu_s * 0.35 + pc_s * 0.20)


# ════════════════════════════════════════════════════════════════════
# LABEL QUALITATIF
# ════════════════════════════════════════════════════════════════════
def label(s):
    return ("Excellent" if s>=.80 else "Good" if s>=.65 else
            "Fair"      if s>=.50 else "Poor" if s>=.35 else "Very Poor")


# ════════════════════════════════════════════════════════════════════
# ASF — ÉVALUATION PAR JOUEUR
#
# Γ(xp) = Σ_D  w̄_D · Φ_D(xp)
#
# Le label ML est attaché en annotation uniquement.
# Aucun facteur PPIF n'est appliqué.
# ════════════════════════════════════════════════════════════════════
def evaluate(p, dw, pdw, tdw, ldw, bdw, ml_level):
    """
    Évalue un joueur selon la méthode ASF.

    Args:
        p        : document joueur (dict)
        dw       : poids des dimensions [PD, TD, LD, BD]
        pdw/tdw/ldw/bdw : poids des critères par dimension
        ml_level : label GMM ('Beginner'|'Intermediate'|'Expert') — annotatif uniquement

    Returns:
        dict avec scores bruts, score global Γ, et label ML
    """
    # ── Scores bruts des critères ────────────────────────────────────
    pdc = {"Ts": f_Ts(p), "Pc": f_Pc(p), "Lr": f_Lr(p), "Em": f_Em(p)}
    tdc = {"Gd": f_Gd(p), "P":  f_P(p),  "Ui": f_Ui(p), "U":  f_U(p)}
    ldc = {"C":  f_C(p),  "F":  f_F(p),  "G":  f_G(p),  "I":  f_I(p)}
    bdc = {"M":  f_M(p),  "E":  f_E(p),  "Ue": f_Ue(p)}

    # ── Scores de dimension Φ_D(xp) ─────────────────────────────────
    PD = sum(pdc[k] * pdw[i] for i, k in enumerate(PD_L))
    TD = sum(tdc[k] * tdw[i] for i, k in enumerate(TD_L))
    LD = sum(ldc[k] * ldw[i] for i, k in enumerate(LD_L))
    BD = sum(bdc[k] * bdw[i] for i, k in enumerate(BD_L))

    # ── Score final ASF : Γ(xp) = Σ w̄_D · Φ_D ────────────────────
    # Aucun PPIF — le label ML n'influe pas sur le calcul numérique
    gamma = PD * dw[0] + TD * dw[1] + LD * dw[2] + BD * dw[3]

    r4 = lambda v: round(v, 4)

    return {
        "player_id"   : p["player_id"],
        # ── Label GMM (annotatif, non-influent sur le score) ─────────
        "ml_level"    : ml_level,
        # ── Critères bruts ───────────────────────────────────────────
        "pd_criteria" : {k: r4(v) for k, v in pdc.items()},
        "td_criteria" : {k: r4(v) for k, v in tdc.items()},
        "ld_criteria" : {k: r4(v) for k, v in ldc.items()},
        "bd_criteria" : {k: r4(v) for k, v in bdc.items()},
        # ── Scores de dimension Φ_D(xp) ──────────────────────────────
        "PD": r4(PD), "TD": r4(TD), "LD": r4(LD), "BD": r4(BD),
        # ── Labels qualitatifs des dimensions ────────────────────────
        "PD_label": label(PD), "TD_label": label(TD),
        "LD_label": label(LD), "BD_label": label(BD),
        # ── Score final Γ(xp) ────────────────────────────────────────
        "global_score": r4(gamma),
        "fuzzy_score" : r4(gamma),   # alias pour compatibilité frontend
        # ── Décision Fuzzy AHP basée sur seuils de Γ ─────────────────
        # (indépendante du label GMM)
        "decision"    : ("expert"       if gamma >= 0.70 else
                         "intermediate" if gamma >= 0.45 else
                         "beginner"),
        # ── Méthode d'intégration ─────────────────────────────────────
        "integration_method": "ASF",
    }


# ════════════════════════════════════════════════════════════════════
# ASF — STATISTIQUES DE GROUPE  (Eq. 2 du PDF)
#
# Γ*_L  = (1/|G_L|) Σ_{p ∈ G_L}  Γ(xp)
# Φ̄_L_D = (1/|G_L|) Σ_{p ∈ G_L}  Φ_D(xp)
# ════════════════════════════════════════════════════════════════════
def compute_group_stats(results):
    """
    Calcule les statistiques agrégées par profil GMM.

    Returns:
        dict { 'beginner': {...}, 'intermediate': {...}, 'expert': {...} }
    """
    profiles = {}
    for lv in ["Beginner", "Intermediate", "Expert"]:
        grp = [r for r in results if r["ml_level"].lower() == lv.lower()]
        if not grp:
            profiles[lv.lower()] = {"count": 0}
            continue

        n = len(grp)
        # Γ*_L — moyenne du score global dans le groupe
        avg_gamma = float(np.mean([r["global_score"] for r in grp]))
        # Φ̄_L_D — moyenne de chaque dimension dans le groupe
        avg_dims  = {d: float(np.mean([r[d] for r in grp])) for d in DIM_L}

        profiles[lv.lower()] = {
            "count"      : n,
            "avg_global" : round(avg_gamma, 4),   # Γ*_L
            **{f"avg_{d}": round(avg_dims[d], 4) for d in DIM_L},  # Φ̄_L_D
        }

    return profiles


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
raw = list(db.players.find({}, {"_id": 0}))
if not raw:
    print("❌ Aucun joueur dans db.players")
    exit(0)

# ── Récupérer les labels GMM depuis players_classified ───────────
ml_map = {
    d["player_id"]: d.get("level", "Intermediate")
    for d in db.players_classified.find({}, {"_id": 0, "player_id": 1, "level": 1})
}

# ── Calculer les poids Fuzzy AHP ─────────────────────────────────
ahp_dim = run_ahp(DIM_M, DIM_L)
ahp_pd  = run_ahp(PD_M,  PD_L)
ahp_td  = run_ahp(TD_M,  TD_L)
ahp_ld  = run_ahp(LD_M,  LD_L)
ahp_bd  = run_ahp(BD_M,  BD_L)

for nm, r in [("DIM",ahp_dim),("PD",ahp_pd),("TD",ahp_td),("LD",ahp_ld),("BD",ahp_bd)]:
    print(f"  {nm}  CR={r['CR']:.4f}  {'✔' if r['consistent'] else '⚠ CR>0.10'}")

dw  = ahp_dim["norm_weights"]
pdw = ahp_pd["norm_weights"]
tdw = ahp_td["norm_weights"]
ldw = ahp_ld["norm_weights"]
bdw = ahp_bd["norm_weights"]

# ── Évaluation ASF de chaque joueur ──────────────────────────────
results = [
    evaluate(p, dw, pdw, tdw, ldw, bdw,
             ml_map.get(p["player_id"], "Intermediate"))
    for p in raw
]

# ── Score global moyen Γ̄ ─────────────────────────────────────────
avgf    = lambda k: float(np.mean([r[k] for r in results]))
avgs    = {d: avgf(d) for d in DIM_L}
gs      = avgf("global_score")

# ── Statistiques de groupe ASF ───────────────────────────────────
group_stats = compute_group_stats(results)

# ── Orientations qualitatives ────────────────────────────────────
dom    = max(avgs, key=avgs.get)
ped    = avgs["PD"]
ent    = (avgs["LD"] + avgs["BD"]) / 2
orient = ("primarily pedagogical" if ped > ent + .05
          else "primarily entertaining" if ent > ped + .05
          else "balanced")

# ── Moyennes des critères ─────────────────────────────────────────
crit_avgs = {}
for dk, ck, cl in [("PD","pd_criteria",PD_L),("TD","td_criteria",TD_L),
                    ("LD","ld_criteria",LD_L),("BD","bd_criteria",BD_L)]:
    crit_avgs[dk] = {
        cr: round(float(np.mean([r[ck][cr] for r in results if cr in r.get(ck,{})])), 4)
        for cr in cl
    }

# ── Suggestions triées par score croissant ────────────────────────
sugg = sorted([
    {"dimension": d, "score": round(avgs[d], 4), "label": label(avgs[d])}
    for d in DIM_L
], key=lambda x: x["score"])

verdict_str = ("Excellent★★★★★" if gs>=.80 else "Good★★★★" if gs>=.65
               else "Average★★★" if gs>=.50 else "Below Average★★" if gs>=.35
               else "Poor★")

# ── Document game_evaluation ──────────────────────────────────────
doc = {
    # Scores moyens des dimensions (scores bruts, sans PPIF)
    "avg_scores"    : {**{d: round(avgs[d], 4) for d in DIM_L}, "global": round(gs, 4)},
    "dim_labels"    : {d: label(avgs[d]) for d in DIM_L},
    "avg_criteria"  : crit_avgs,

    # Statistiques de groupe ASF — Eq. 2 du PDF
    "by_player_type": group_stats,

    "suggestions"   : sugg,
    "verdict"       : {
        "rating"       : verdict_str,
        "satisfied"    : gs >= .50,
        "learning"     : gs >= .65,
        "orientation"  : orient,
        "dominant_dim" : dom,
        "summary"      : f"Score global Γ̄={gs:.3f} — {label(gs)} (méthode ASF)",
    },
    "dim_weights"   : {DIM_L[i]: round(dw[i], 4) for i in range(4)},

    # Méthode d'intégration ML–Fuzzy
    "integration"   : {
        "method"      : "Average Score Fusion (ASF)",
        "ml_model"    : "GMM",
        "description" : (
            "Le label GMM (Beginner/Intermediate/Expert) est attaché à chaque "
            "joueur à titre descriptif. Il n'influe pas sur le calcul du score "
            "Fuzzy AHP : Γ(xp) = Σ_D w̄_D · Φ_D(xp)."
        ),
    },

    "n_players"     : len(results),
    "timestamp"     : time.time(),
}

db.game_evaluation.delete_many({})
db.game_evaluation.insert_one(doc)
db.fuzzy_results.delete_many({})
db.fuzzy_results.insert_many(results)

print(f"✅ ASF Fuzzy AHP — {len(results)} joueurs — Γ̄={gs:.4f} — {verdict_str}")
print(f"   Poids dimensions : {dict(zip(DIM_L, [round(w,4) for w in dw]))}")
print(f"   Groupes : { {k: v.get('count',0) for k,v in group_stats.items()} }")