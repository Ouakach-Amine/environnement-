import os
import time
import joblib
import numpy as np
import pandas as pd

from pymongo import MongoClient
from collections import Counter

# ==========================================================
# MongoDB
# ==========================================================

client = MongoClient("mongodb://localhost:27017/")
db = client["game_db"]

# ==========================================================
# Charger modèle
# ==========================================================

PKL_PATH = "player_classification_gmm_3.pkl"

print(f"📦 Chargement : {PKL_PATH}")

model_data = joblib.load(PKL_PATH)

FEATURES = model_data["features"]
NUMERIC_FEATURES = model_data["numeric_features"]
DEVICE_COLUMNS = model_data["device_columns"]

imputer = model_data["imputer"]
scaler = model_data["scaler"]
gmm = model_data["gmm"]

cluster_mapping = model_data["cluster_mapping"]

sil_training = model_data["silhouette_score"]

print("✅ Modèle chargé")
print("Features:", len(FEATURES))
print("Clusters:", gmm.n_components)
print("Mapping:", cluster_mapping)
print("Silhouette:", round(sil_training, 4))

# ==========================================================
# Préparation vecteur
# ==========================================================

def prepare_dataframe(player):

    data = {}

    # features numériques
    for f in NUMERIC_FEATURES:

        val = player.get(f, 0)

        try:
            data[f] = float(val)
        except:
            data[f] = 0.0

    # device_type
    device = str(
        player.get("device_type", "desktop")
    ).lower()

    for col in DEVICE_COLUMNS:

        if col == f"device_type_{device}":
            data[col] = 1
        else:
            data[col] = 0

    return pd.DataFrame([data])

# ==========================================================
# Prediction
# ==========================================================

def predict_player(player):

    df = prepare_dataframe(player)

    # ordre exact
    df = df.reindex(columns=FEATURES, fill_value=0)

    # NaN handling
    X_imputed = imputer.transform(df)

    # scaling
    X_scaled = scaler.transform(X_imputed)

    # cluster
    cluster = gmm.predict(X_scaled)[0]

    # probabilités
    probs = gmm.predict_proba(X_scaled)[0]

    confidence = round(float(np.max(probs)), 4)

    level = cluster_mapping.get(
        str(cluster),
        "Intermediate"
    )

    return {
        **player,
        "cluster": int(cluster),
        "level": level,
        "confidence": confidence,
        "model": "GaussianMixture"
    }

# ==========================================================
# Pipeline
# ==========================================================

print("\n🔄 Classification des joueurs...")

raw_players = list(
    db.players.find({}, {"_id": 0})
)

print(f"   {len(raw_players)} joueurs trouvés")

if len(raw_players) == 0:
    print("Aucun joueur")
    exit()

classified = []

errors = 0

for player in raw_players:

    try:

        result = predict_player(player)

        classified.append(result)

    except Exception as e:

        print(
            f"⚠ Joueur #{player.get('player_id')}: {e}"
        )

        errors += 1

# ==========================================================
# Distribution
# ==========================================================

dist = Counter(
    p["level"] for p in classified
)

print("\nDistribution:")
print(dict(dist))

# ==========================================================
# Sauvegarde MongoDB
# ==========================================================

db.players_classified.delete_many({})

if classified:
    db.players_classified.insert_many(classified)

print(
    f"\n✅ players_classified → {len(classified)} docs"
)

# ==========================================================
# Métriques
# ==========================================================

db.model_metrics.delete_many({})

db.model_metrics.insert_one({

    "model": "GaussianMixture",

    "n_clusters": int(gmm.n_components),

    "features": len(FEATURES),

    "cluster_mapping": {
        str(k): str(v)
        for k, v in cluster_mapping.items()
    },

    "silhouette_score": float(sil_training),

    "distribution": dict(dist),

    "players": len(classified),

    "errors": errors,

    "timestamp": time.time()
})

print("✅ model_metrics sauvegardé")