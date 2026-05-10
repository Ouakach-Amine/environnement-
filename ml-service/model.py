"""
ML Service — KMeans / DBSCAN / Agglomerative + Silhouette + DBI
Writes to: db.players_classified  and  db.model_metrics
"""
import time
import traceback
from collections import Counter

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════
# MongoDB connection with retry
# ══════════════════════════════════════════════════════════════════
def get_db(retries=30, delay=5):
    for i in range(1, retries + 1):
        try:
            client = MongoClient("mongodb://mongodb:27017/",
                                 serverSelectionTimeoutMS=3000)
            client.server_info()
            print(f"✅ MongoDB connected (attempt {i})")
            return client["game_db"]
        except Exception as e:
            print(f"⏳ MongoDB not ready [{i}/{retries}]: {e}")
            time.sleep(delay)
    raise RuntimeError("Cannot connect to MongoDB")


db = get_db()


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════
def safe_metrics(X, labels):
    """
    Compute (silhouette, DBI) safely.
    - Filters DBSCAN noise points (label == -1)
    - Returns (-1, inf) for trivial / degenerate clusterings
    """
    mask = labels != -1
    Xf, yf = X[mask], labels[mask]
    if len(set(yf)) < 2 or len(Xf) < 2:
        return -1.0, float("inf")
    try:
        sil = float(silhouette_score(Xf, yf))
        dbi = float(davies_bouldin_score(Xf, yf))
        return sil, dbi
    except Exception as e:
        print(f"  ⚠ metric error: {e}")
        return -1.0, float("inf")


def map_level(labels, scores):
    """
    Rank clusters by mean score → beginner / intermediate / expert.
    Noise points (label=-1) → intermediate.
    """
    df_tmp = pd.DataFrame({"cluster": labels, "score": scores})
    means  = (df_tmp.groupby("cluster")["score"]
              .mean()
              .drop(index=-1, errors="ignore")
              .sort_values())
    valid  = list(means.index)

    mapping = {-1: "intermediate"}
    for i, c in enumerate(valid):
        if len(valid) == 1:
            mapping[c] = "intermediate"
        elif i == 0:
            mapping[c] = "beginner"
        elif i == len(valid) - 1:
            mapping[c] = "expert"
        else:
            mapping[c] = "intermediate"

    return [mapping.get(int(l), "intermediate") for l in labels]


# ══════════════════════════════════════════════════════════════════
# Main pipeline loop
# ══════════════════════════════════════════════════════════════════
print("🚀 ML Service started")

while True:
    try:
        print("\n" + "─" * 55)
        print("🔄 Running ML pipeline...")

        # 1. Load players
        raw = list(db.players.find({}, {"_id": 0}))
        print(f"   players: {len(raw)} records")

        if len(raw) < 5:
            print("⏳ Need ≥5 players — waiting 10 s")
            time.sleep(10)
            continue

        df = pd.DataFrame(raw)

        # 2. Feature selection
        feat = df.select_dtypes(include=["int64", "float64",
                                          "int32", "float32"])
        feat = feat.dropna(axis=1, how="all")
        feat = feat.loc[:, feat.std() > 0]          # drop constant cols

        print(f"   Features: {list(feat.columns)}")

        if feat.shape[1] < 2:
            print("❌ Fewer than 2 usable features — waiting 10 s")
            time.sleep(10)
            continue

        # 3. Standardise
        X = StandardScaler().fit_transform(feat)
        n = len(X)
        k = min(3, n - 1)

        if k < 2:
            print("❌ n < 3 samples — waiting 10 s")
            time.sleep(10)
            continue

        # 4. Fit three models
        km_labels = KMeans(n_clusters=k, random_state=42,
                           n_init=10).fit_predict(X)
        db_labels = DBSCAN(eps=1.5, min_samples=3).fit_predict(X)
        ag_labels = AgglomerativeClustering(
                        n_clusters=k).fit_predict(X)

        label_map = {"kmeans": km_labels,
                     "dbscan": db_labels,
                     "agglo":  ag_labels}

        # 5. Metrics
        m = {}
        for name, lbls in label_map.items():
            sil, dbi = safe_metrics(X, lbls)
            m[name] = {"sil": sil, "dbi": dbi}
            dbi_str = f"{dbi:.4f}" if dbi != float("inf") else "∞"
            print(f"   {name:<12} Silhouette={sil:+.4f}  DBI={dbi_str}")

        # 6. Composite selection score
        finite_dbis = [v["dbi"] for v in m.values()
                       if v["dbi"] != float("inf")]
        max_dbi = max(finite_dbis) if finite_dbis else 1.0

        def composite(info):
            s, d = info["sil"], info["dbi"]
            if s == -1 or d == float("inf"):
                return -999.0
            return s + (1.0 - d / max_dbi)

        best = max(m, key=lambda nm: composite(m[nm]))
        final_labels = label_map[best]
        print(f"   🏆 Best: {best.upper()}  "
              f"(composite={composite(m[best]):.4f})")

        # 7. Level mapping
        score_vals = (df["score"].values
                      if "score" in df.columns
                      else np.zeros(n))
        levels = map_level(final_labels, score_vals)
        print(f"   Distribution: {dict(Counter(levels))}")

        # 8. Build classified docs
        classified = []
        for i, row in df.iterrows():
            doc          = row.to_dict()
            doc["cluster"] = int(final_labels[i])
            doc["level"]   = levels[i]
            doc["model"]   = best
            classified.append(doc)

        # ── Write players_classified ──────────────────────────────
        db.players_classified.delete_many({})
        result = db.players_classified.insert_many(classified)
        print(f"✅ players_classified → {len(result.inserted_ids)} docs")

        # ── Write model_metrics ───────────────────────────────────
        def to_val(d):
            return round(d, 4) if d != float("inf") else -1.0

        metrics_doc = {
            # Silhouette (higher = better)
            "sil_kmeans":  round(m["kmeans"]["sil"], 4),
            "sil_dbscan":  round(m["dbscan"]["sil"], 4),
            "sil_agglo":   round(m["agglo"]["sil"],  4),
            # Davies-Bouldin Index (lower = better, 0 = ideal)
            "dbi_kmeans":  to_val(m["kmeans"]["dbi"]),
            "dbi_dbscan":  to_val(m["dbscan"]["dbi"]),
            "dbi_agglo":   to_val(m["agglo"]["dbi"]),
            # Composite
            "comp_kmeans": round(composite(m["kmeans"]), 4),
            "comp_dbscan": round(composite(m["dbscan"]), 4),
            "comp_agglo":  round(composite(m["agglo"]),  4),
            # Winner + meta
            "best_model":  best,
            "n_players":   len(classified),
            "timestamp":   time.time(),
        }

        db.model_metrics.delete_many({})
        db.model_metrics.insert_one(metrics_doc)
        print(f"✅ model_metrics → best={best.upper()}")

    except Exception:
        print("❌ Pipeline error:")
        traceback.print_exc()

    time.sleep(15)
