"""
ML Service — runs directly on the host machine (no Docker)
Connects to MongoDB localhost:27017
"""
import time, traceback
from collections import Counter

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def get_db(retries=30, delay=5):
    for i in range(1, retries + 1):
        try:
            client = MongoClient("mongodb://localhost:27017/",
                                 serverSelectionTimeoutMS=3000)
            client.server_info()
            print(f"✅ MongoDB connected (attempt {i})")
            return client["game_db"]
        except Exception as e:
            print(f"⏳ [{i}/{retries}] {e}")
            time.sleep(delay)
    raise RuntimeError("Cannot connect to MongoDB")

db = get_db()


def safe_metrics(X, labels):
    mask = labels != -1
    Xf, yf = X[mask], labels[mask]
    if len(set(yf)) < 2 or len(Xf) < 2:
        return -1.0, float("inf")
    try:
        return float(silhouette_score(Xf, yf)), float(davies_bouldin_score(Xf, yf))
    except:
        return -1.0, float("inf")


def map_level(labels, scores):
    df_tmp = pd.DataFrame({"cluster": labels, "score": scores})
    means  = (df_tmp.groupby("cluster")["score"]
              .mean().drop(index=-1, errors="ignore").sort_values())
    valid  = list(means.index)
    mapping = {-1: "intermediate"}
    for i, c in enumerate(valid):
        if len(valid) == 1:        mapping[c] = "intermediate"
        elif i == 0:               mapping[c] = "beginner"
        elif i == len(valid) - 1:  mapping[c] = "expert"
        else:                      mapping[c] = "intermediate"
    return [mapping.get(int(l), "intermediate") for l in labels]


print("🚀 ML Service started (localhost mode)")

while True:
    try:
        print("\n" + "─" * 50)
        raw = list(db.players.find({}, {"_id": 0}))
        print(f"   players: {len(raw)}")

        if len(raw) < 5:
            print("⏳ Need ≥5 players"); time.sleep(10); continue

        df   = pd.DataFrame(raw)
        feat = df.select_dtypes(include=["int64","float64","int32","float32"])
        feat = feat.dropna(axis=1, how="all")
        feat = feat.loc[:, feat.std() > 0]
        if feat.shape[1] < 2:
            print("❌ Not enough features"); time.sleep(10); continue

        X = StandardScaler().fit_transform(feat)
        n = len(X); k = min(3, n - 1)
        if k < 2:
            print("❌ Not enough samples"); time.sleep(10); continue

        km_l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        db_l = DBSCAN(eps=1.5, min_samples=3).fit_predict(X)
        ag_l = AgglomerativeClustering(n_clusters=k).fit_predict(X)

        label_map = {"kmeans": km_l, "dbscan": db_l, "agglo": ag_l}
        m = {}
        for name, lbls in label_map.items():
            sil, dbi = safe_metrics(X, lbls)
            m[name]  = {"sil": sil, "dbi": dbi}
            dbi_str  = "∞" if dbi == float("inf") else f"{dbi:.4f}"
            print(f"   {name:<12} Sil={sil:+.4f}  DBI={dbi_str}")

        finite  = [v["dbi"] for v in m.values() if v["dbi"] != float("inf")]
        max_dbi = max(finite) if finite else 1.0

        def composite(info):
            s, d = info["sil"], info["dbi"]
            return -999.0 if (s == -1 or d == float("inf")) else s + (1 - d / max_dbi)

        best         = max(m, key=lambda nm: composite(m[nm]))
        final_labels = label_map[best]
        score_vals   = df["score"].values if "score" in df.columns else np.zeros(n)
        levels       = map_level(final_labels, score_vals)

        print(f"   🏆 {best.upper()}  dist={dict(Counter(levels))}")

        classified = []
        for i, row in df.iterrows():
            doc = row.to_dict()
            doc["cluster"] = int(final_labels[i])
            doc["level"]   = levels[i]
            doc["model"]   = best
            classified.append(doc)

        db.players_classified.delete_many({})
        db.players_classified.insert_many(classified)

        def to_val(d): return round(d, 4) if d != float("inf") else -1.0

        db.model_metrics.delete_many({})
        db.model_metrics.insert_one({
            "sil_kmeans" : round(m["kmeans"]["sil"], 4),
            "sil_dbscan" : round(m["dbscan"]["sil"], 4),
            "sil_agglo"  : round(m["agglo"]["sil"],  4),
            "dbi_kmeans" : to_val(m["kmeans"]["dbi"]),
            "dbi_dbscan" : to_val(m["dbscan"]["dbi"]),
            "dbi_agglo"  : to_val(m["agglo"]["dbi"]),
            "comp_kmeans": round(composite(m["kmeans"]), 4),
            "comp_dbscan": round(composite(m["dbscan"]), 4),
            "comp_agglo" : round(composite(m["agglo"]),  4),
            "best_model" : best,
            "n_players"  : len(classified),
            "timestamp"  : time.time(),
        })
        print(f"✅ Saved {len(classified)} classified + metrics")

    except Exception:
        print("❌ Error:"); traceback.print_exc()

    time.sleep(15)