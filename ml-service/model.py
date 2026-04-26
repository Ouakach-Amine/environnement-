from pymongo import MongoClient
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Connexion MongoDB
client = MongoClient("mongodb://mongodb:27017/")
db = client["game_db"]

# Lire données
data = list(db.players.find({}, {"_id": 0}))

if len(data) == 0:
    print("❌ No data found")
    exit()

df = pd.DataFrame(data)

print("✅ Data loaded:", df.shape)

# --------------------------
# 🔷 1. Préparation des features
# --------------------------

features = df.select_dtypes(include=['int64', 'float64'])

scaler = StandardScaler()
X = scaler.fit_transform(features)

# --------------------------
# 🔷 2. K-Means
# --------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X)

score_kmeans = silhouette_score(X, labels_kmeans)
print("KMeans score:", score_kmeans)

# --------------------------
# 🔷 3. DBSCAN
# --------------------------
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

# vérifier bruit (-1)
if len(set(labels_dbscan)) > 1:
    score_dbscan = silhouette_score(X, labels_dbscan)
else:
    score_dbscan = -1

print("DBSCAN score:", score_dbscan)

# --------------------------
# 🔷 4. Clustering hiérarchique
# --------------------------
agglo = AgglomerativeClustering(n_clusters=3)
labels_agglo = agglo.fit_predict(X)

score_agglo = silhouette_score(X, labels_agglo)
print("Agglomerative score:", score_agglo)

# --------------------------
# 🔷 5. Comparaison
# --------------------------
scores = {
    "kmeans": score_kmeans,
    "dbscan": score_dbscan,
    "agglo": score_agglo
}

best_model = max(scores, key=scores.get)

print("🏆 Best model:", best_model)

# --------------------------
# 🔷 6. Mapper clusters → niveau joueur
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

# appliquer meilleur modèle
if best_model == "kmeans":
    final_labels = labels_kmeans
elif best_model == "dbscan":
    final_labels = labels_dbscan
else:
    final_labels = labels_agglo

levels = map_level(final_labels, df)

# --------------------------
# 🔷 7. Sauvegarde MongoDB
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

# Sauvegarder métriques
db.model_metrics.delete_many({})
db.model_metrics.insert_one({
    "kmeans": score_kmeans,
    "dbscan": score_dbscan,
    "agglo": score_agglo,
    "best_model": best_model
})

print("✅ Results saved to MongoDB")
