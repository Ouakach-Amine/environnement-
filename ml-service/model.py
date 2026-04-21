from sklearn.cluster import KMeans
import numpy as np

# exemple simple
data = np.array([
    [10, 2],   # joueur 1
    [50, 10],  # joueur 2
    [80, 20]
])

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

print(kmeans.labels_)
