"""
Fuzzy AHP — runs on host machine, connects to MongoDB localhost
"""
import time, math
import numpy as np
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db     = client["game_db"]

# ── TFN algebra ───────────────────────────────────────────────────
def tfn_add(tfns):
    return (sum(t[0] for t in tfns), sum(t[1] for t in tfns), sum(t[2] for t in tfns))
def tfn_mul(a,b): return (a[0]*b[0], a[1]*b[1], a[2]*b[2])
def tfn_recip(a): return (1/a[2], 1/a[1], 1/a[0])
def tfn_geo(row):
    n=len(row)
    return (math.prod(t[0] for t in row)**(1/n),
            math.prod(t[1] for t in row)**(1/n),
            math.prod(t[2] for t in row)**(1/n))
def defuzz(t): return (t[0]+t[1]+t[2])/3

_SC = {1:(1,1,1),2:(1,2,3),3:(2,3,4),4:(3,4,5),5:(4,5,6),
       6:(5,6,7),7:(6,7,8),8:(7,8,9),9:(8,9,9)}
def sc(v):
    return _SC[int(round(v))] if v>=1 else tfn_recip(_SC[int(round(1/v))])

RI = {1:0,2:0,3:.58,4:.90,5:1.12,6:1.24,7:1.32,8:1.41}

def run_ahp(M, labels):
    n  = len(M)
    Mf = [[sc(M[i][j]) for j in range(n)] for i in range(n)]
    Mm = np.array([[Mf[i][j][1] for j in range(n)] for i in range(n)],dtype=float)
    w  = (Mm/Mm.sum(axis=0)).mean(axis=1)
    lm = float(np.mean((Mm@w)/w))
    ci = (lm-n)/(n-1) if n>1 else 0
    cr = ci/RI.get(n,1.49) if RI.get(n,1)>0 else 0
    gm = [tfn_geo(Mf[i]) for i in range(n)]
    ti = tfn_recip(tfn_add(gm))
    fw = [tfn_mul(g,ti) for g in gm]
    cw = [defuzz(f) for f in fw]
    s  = sum(cw)
    nw = [c/s for c in cw]
    td = lambda t:{"l":round(t[0],4),"m":round(t[1],4),"u":round(t[2],4)}
    return {"labels":labels,"lambda_max":round(lm,4),"CI":round(ci,4),
            "CR":round(cr,4),"consistent":bool(cr<=0.10),
            "geo_means":[td(g) for g in gm],"fuzzy_weights":[td(f) for f in fw],
            "norm_weights":[round(w,4) for w in nw],
            "weight_map":{labels[i]:round(nw[i],4) for i in range(n)}}

# ── Matrices ──────────────────────────────────────────────────────
DIM_M=[[1,3,5,7],[1/3,1,3,5],[1/5,1/3,1,3],[1/7,1/5,1/3,1]]
PD_M =[[1,2,1/2,3],[1/2,1,1/3,2],[2,3,1,5],[1/3,1/2,1/5,1]]
TD_M =[[1,1/2,2,1/2],[2,1,3,2],[1/2,1/3,1,1/2],[2,1/2,2,1]]
LD_M =[[1,1,2,3],[1,1,2,3],[1/2,1/2,1,2],[1/3,1/3,1/2,1]]
BD_M =[[1,2,3],[1/2,1,2],[1/3,1/2,1]]
DIM_L=["PD","TD","LD","BD"]
PD_L =["Ts","Pc","Lr","Em"]
TD_L =["Gd","P","Ui","U"]
LD_L =["C","F","G","I"]
BD_L =["M","E","Ue"]

# ── Membership functions ──────────────────────────────────────────
clamp=lambda v:max(0.,min(1.,float(v)))
def f_Ts(p): return clamp(clamp(p.get("level",1)/5)*.55+float(p.get("success",0))*.45)
def f_Pc(p): return clamp(p.get("score",0)/max(p.get("level",1)*20.,1))
def f_Lr(p): return clamp(p.get("score",0)/100*.65+float(p.get("success",0))*.35)
def f_Em(p):
    e=p.get("errors",0)
    return 1. if e==0 else .85 if e<=2 else .60 if e<=5 else .35 if e<=9 else .15
def f_Gd(p): return clamp((p.get("clicks",0)+p.get("moves",0))/80)
def f_P(p):
    r=p.get("response_time",0)
    return 1. if r<=.3 else .9 if r<=.8 else .75 if r<=1.5 else .5 if r<=3 else .3 if r<=5 else .1
def f_Ui(p):
    c,mv=p.get("clicks",0),p.get("moves",0); t=c+mv
    return .5 if t==0 else clamp(1-abs(c/t-.5)*1.6)
def f_U(p):  return clamp(f_Em(p)*.5+f_P(p)*.5)
def f_C(p):  return clamp(p.get("level",1)/5*.5+p.get("score",0)/100*.5)
def f_F(p):
    r=p.get("repetition",0)
    return .3 if r==0 else .7 if r<=2 else 1. if r<=4 else .75 if r<=7 else .45
def f_G(p):  return clamp((p.get("moves",0)+p.get("clicks",0))/100)
def f_I(p):
    t=p.get("time",0)
    ts=1. if 15<=t<=45 else .7 if(10<=t<15 or 45<t<=70) else .3 if t<10 else .4
    return clamp(ts*.6+clamp(p.get("repetition",0)/4)*.4)
def f_M(p):  return clamp(p.get("repetition",0)/5*.55+float(p.get("success",0))*.45)
def f_E(p):  return clamp(clamp(p.get("time",0)/60)*.4+clamp((p.get("clicks",0)+p.get("moves",0))/80)*.6)
def f_Ue(p): return clamp(f_Em(p)*.35+float(p.get("success",0))*.35+p.get("score",0)/100*.3)

PPIF={"beginner":{"PD":.82,"TD":.90,"LD":1.28,"BD":1.20},
      "intermediate":{"PD":1.,"TD":1.,"LD":1.,"BD":1.},
      "expert":{"PD":1.22,"TD":1.16,"LD":.82,"BD":.88}}

def label(s):
    return("Excellent"if s>=.8 else"Good"if s>=.65 else"Fair"if s>=.5 else"Poor"if s>=.35 else"Very Poor")

def evaluate(p,dw,pdw,tdw,ldw,bdw,ml):
    pdc={"Ts":f_Ts(p),"Pc":f_Pc(p),"Lr":f_Lr(p),"Em":f_Em(p)}
    tdc={"Gd":f_Gd(p),"P":f_P(p),"Ui":f_Ui(p),"U":f_U(p)}
    ldc={"C":f_C(p),"F":f_F(p),"G":f_G(p),"I":f_I(p)}
    bdc={"M":f_M(p),"E":f_E(p),"Ue":f_Ue(p)}
    PD=sum(pdc[k]*pdw[i] for i,k in enumerate(PD_L))
    TD=sum(tdc[k]*tdw[i] for i,k in enumerate(TD_L))
    LD=sum(ldc[k]*ldw[i] for i,k in enumerate(LD_L))
    BD=sum(bdc[k]*bdw[i] for i,k in enumerate(BD_L))
    IF=PPIF.get((ml or"intermediate").lower(),PPIF["intermediate"])
    Pa,Ta,La,Ba=clamp(PD*IF["PD"]),clamp(TD*IF["TD"]),clamp(LD*IF["LD"]),clamp(BD*IF["BD"])
    gs=Pa*dw[0]+Ta*dw[1]+La*dw[2]+Ba*dw[3]
    r4=lambda v:round(v,4)
    return{"player_id":p["player_id"],"ml_level":ml,
           "pd_criteria":{k:r4(v) for k,v in pdc.items()},
           "td_criteria":{k:r4(v) for k,v in tdc.items()},
           "ld_criteria":{k:r4(v) for k,v in ldc.items()},
           "bd_criteria":{k:r4(v) for k,v in bdc.items()},
           "PD":r4(PD),"TD":r4(TD),"LD":r4(LD),"BD":r4(BD),
           "PD_adj":r4(Pa),"TD_adj":r4(Ta),"LD_adj":r4(La),"BD_adj":r4(Ba),
           "PD_label":label(Pa),"TD_label":label(Ta),"LD_label":label(La),"BD_label":label(Ba),
           "global_score":r4(gs),"fuzzy_score":r4(gs)}

# ── Main ──────────────────────────────────────────────────────────
raw=list(db.players.find({},{"_id":0}))
if not raw: print("❌ No data"); exit(1)
ml_map={d["player_id"]:d.get("level","intermediate")
        for d in db.players_classified.find({},{"_id":0,"player_id":1,"level":1})}

ahp_dim=run_ahp(DIM_M,DIM_L); ahp_pd=run_ahp(PD_M,PD_L)
ahp_td=run_ahp(TD_M,TD_L);   ahp_ld=run_ahp(LD_M,LD_L)
ahp_bd=run_ahp(BD_M,BD_L)

for nm,r in[("DIM",ahp_dim),("PD",ahp_pd),("TD",ahp_td),("LD",ahp_ld),("BD",ahp_bd)]:
    print(f"  {nm}  CR={r['CR']:.4f}  {'✔' if r['consistent'] else '⚠'}")

dw=ahp_dim["norm_weights"]
results=[evaluate(p,dw,ahp_pd["norm_weights"],ahp_td["norm_weights"],
                  ahp_ld["norm_weights"],ahp_bd["norm_weights"],
                  ml_map.get(p["player_id"],"intermediate")) for p in raw]

avgf=lambda k:float(np.mean([r[k] for r in results]))
avgs={"PD":avgf("PD_adj"),"TD":avgf("TD_adj"),"LD":avgf("LD_adj"),"BD":avgf("BD_adj")}
gs=avgf("global_score")

dom=max(avgs,key=avgs.get)
ped=avgs["PD"]; ent=(avgs["LD"]+avgs["BD"])/2
orient=("primarily pedagogical"if ped>ent+.05 else
        "primarily entertaining"if ent>ped+.05 else"balanced")

profiles={}
for lv in["beginner","intermediate","expert"]:
    grp=[r for r in results if r["ml_level"]==lv]
    if not grp: profiles[lv]={"count":0}; continue
    profiles[lv]={"count":len(grp),
        "avg_global":round(float(np.mean([r["global_score"]for r in grp])),4),
        "avg_PD":round(float(np.mean([r["PD_adj"]for r in grp])),4),
        "avg_TD":round(float(np.mean([r["TD_adj"]for r in grp])),4),
        "avg_LD":round(float(np.mean([r["LD_adj"]for r in grp])),4),
        "avg_BD":round(float(np.mean([r["BD_adj"]for r in grp])),4)}

crit_avgs={}
for dk,ck,cl in[("PD","pd_criteria",PD_L),("TD","td_criteria",TD_L),
                ("LD","ld_criteria",LD_L),("BD","bd_criteria",BD_L)]:
    crit_avgs[dk]={c:round(float(np.mean([r[ck][c]for r in results if c in r.get(ck,{})])),4)for c in cl}

sugg=[{"dimension":d,"score":round(avgs[d],4),"label":label(avgs[d])} for d in DIM_L]
sugg.sort(key=lambda x:x["score"])

doc={"avg_scores":{"PD":round(avgs["PD"],4),"TD":round(avgs["TD"],4),
                   "LD":round(avgs["LD"],4),"BD":round(avgs["BD"],4),"global":round(gs,4)},
     "dim_labels":{"PD":label(avgs["PD"]),"TD":label(avgs["TD"]),
                   "LD":label(avgs["LD"]),"BD":label(avgs["BD"]),"global":label(gs)},
     "avg_criteria":crit_avgs,"by_player_type":profiles,"suggestions":sugg,
     "verdict":{"rating":("Excellent★★★★★"if gs>=.8 else"Good★★★★"if gs>=.65 else
                          "Average★★★"if gs>=.5 else"Below Average★★"if gs>=.35 else"Poor★"),
                "satisfied":gs>=.5,"learning":gs>=.65,"orientation":orient,
                "dominant_dim":dom,"summary":f"Global score {gs:.3f} — {label(gs)}"},
     "n_players":len(results),"timestamp":time.time()}

db.game_evaluation.delete_many({}); db.game_evaluation.insert_one(doc)
db.fuzzy_results.delete_many({}); db.fuzzy_results.insert_many(results)
print(f"✅ Done — {len(results)} players — global={gs:.4f}")