from pymongo import MongoClient
import pandas as pd
import time
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Connexion MongoDB
client = MongoClient("mongodb://mongodb:27017/")
db = client["game_db"]

print("🚀 ML Service started...")

while True:
    try:
        print("\n🔄 Running ML pipeline...")

        # --------------------------
        # 🔷 1. Charger données
        # --------------------------
        data = list(db.players.find({}, {"_id": 0}))

        if len(data) < 5:
            print("⏳ Not enough data (min 5), waiting...")
            time.sleep(10)
            continue

        df = pd.DataFrame(data)
        print("✅ Data loaded:", df.shape)

        # --------------------------
        # 🔷 2. Features
        # --------------------------
        features = df.select_dtypes(include=['int64', 'float64'])

        if features.shape[1] < 2:
            print("❌ Not enough features")
            time.sleep(10)
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        n_samples = len(X)
        k = min(3, n_samples - 1)

        if k < 2:
            print("❌ Not enough samples for clustering")
            time.sleep(10)
            continue

        # --------------------------
        # 🔷 3. KMeans
        # --------------------------
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels_kmeans = kmeans.fit_predict(X)

        if len(set(labels_kmeans)) > 1:
            score_kmeans = silhouette_score(X, labels_kmeans)
        else:
            score_kmeans = -1

        print("KMeans score:", score_kmeans)

        # --------------------------
        # 🔷 4. DBSCAN
        # --------------------------
        dbscan = DBSCAN(eps=1.5, min_samples=3)
        labels_dbscan = dbscan.fit_predict(X)

        valid_labels = set(labels_dbscan) - {-1}

        if len(valid_labels) > 1:
            score_dbscan = silhouette_score(X, labels_dbscan)
        else:
            score_dbscan = -1

        print("DBSCAN score:", score_dbscan)

        # --------------------------
        # 🔷 5. Agglomerative
        # --------------------------
        agglo = AgglomerativeClustering(n_clusters=k)
        labels_agglo = agglo.fit_predict(X)

        if len(set(labels_agglo)) > 1:
            score_agglo = silhouette_score(X, labels_agglo)
        else:
            score_agglo = -1

        print("Agglomerative score:", score_agglo)

        # --------------------------
        # 🔷 6. Comparaison
        # --------------------------
        scores = {
            "kmeans": score_kmeans,
            "dbscan": score_dbscan,
            "agglo": score_agglo
        }

        best_model = max(scores, key=scores.get)

        print("🏆 Best model:", best_model)

        # --------------------------
        # 🔷 7. Sélection labels
        # --------------------------
        if best_model == "kmeans":
            final_labels = labels_kmeans
        elif best_model == "dbscan":
            final_labels = labels_dbscan
        else:
            final_labels = labels_agglo

        # --------------------------
        # 🔷 8. Mapping niveau joueur
        # --------------------------
        def map_level(labels, df):
            df_temp = df.copy()
            df_temp["cluster"] = labels

            means = df_temp.groupby("cluster")["score"].mean().sort_values()

            mapping = {}
            for i, cluster in enumerate(means.index):
                if i == 0:
                    mapping[cluster] = "beginner"
                elif i == 1:
                    mapping[cluster] = "intermediate"
                else:
                    mapping[cluster] = "expert"

            return [mapping[l] for l in labels]

        levels = map_level(final_labels, df)

        # --------------------------
        # 🔷 9. Sauvegarde MongoDB
        # --------------------------
        results = []

        for i, row in df.iterrows():
            doc = row.to_dict()
            doc["cluster"] = int(final_labels[i])
            doc["level"] = levels[i]
            doc["model"] = best_model
            results.append(doc)

        db.players_classified.delete_many({})
        db.players_classified.insert_many(results)

        db.model_metrics.delete_many({})
        db.model_metrics.insert_one({
            "kmeans": float(score_kmeans),
            "dbscan": float(score_dbscan),
            "agglo": float(score_agglo),
            "best_model": best_model,
            "timestamp": time.time()
        })

        print("✅ Results saved to MongoDB")

    except Exception as e:
        print("❌ Error:", str(e))

    # attendre avant prochain cycle
    time.sleep(15)
