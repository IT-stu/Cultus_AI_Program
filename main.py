# ============================================================
# K-Means Clustering From Scratch Using NumPy
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# ------------------------------------------------------------
# 1. Generate Synthetic 2D Dataset (3 non-uniform clusters)
# ------------------------------------------------------------
X, y_true = make_blobs(
    n_samples=[100, 150, 80],
    centers=[(-5, -5), (0, 0), (5, 5)],
    cluster_std=[1.0, 2.5, 0.8],
    random_state=42
)

plt.scatter(X[:, 0], X[:, 1])
plt.title("Synthetic Dataset with 3 Non-Uniform Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# ------------------------------------------------------------
# 2. K-Means Implementation from Scratch (NumPy only)
# ------------------------------------------------------------

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def random_init(X, k):
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx]

def kmeans_plus_plus(X, k):
    centroids = [X[np.random.randint(len(X))]]

    for _ in range(1, k):
        distances = np.array([
            min(euclidean_distance(x, c) for c in centroids)
            for x in X
        ])
        probs = distances / distances.sum()
        cumulative_probs = np.cumsum(probs)
        r = np.random.rand()

        for i, p in enumerate(cumulative_probs):
            if r < p:
                centroids.append(X[i])
                break

    return np.array(centroids)

def kmeans(X, k, init="random", max_iters=100):
    if init == "random":
        centroids = random_init(X, k)
    else:
        centroids = kmeans_plus_plus(X, k)

    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]

        for point in X:
            distances = [euclidean_distance(point, c) for c in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)

        new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters

# ------------------------------------------------------------
# 3. Elbow Method (K = 1 to 10)
# ------------------------------------------------------------

def inertia(X, centroids, clusters):
    total = 0
    for i, cluster in enumerate(clusters):
        for point in cluster:
            total += euclidean_distance(point, centroids[i]) ** 2
    return total

inertias = []

for k in range(1, 11):
    centroids, clusters = kmeans(X, k)
    inertias.append(inertia(X, centroids, clusters))

plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# ------------------------------------------------------------
# 4. Run K-Means with Optimal K (K = 3)
#    Compare Random vs K-Means++
# ------------------------------------------------------------

k = 3

centroids_random, clusters_random = kmeans(X, k, init="random")
centroids_pp, clusters_pp = kmeans(X, k, init="kmeans++")

# ------------------------------------------------------------
# 5. Visualization of Final Clusters
# ------------------------------------------------------------

def plot_clusters(clusters, centroids, title):
    for cluster in clusters:
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1],
                c='black', marker='X', s=200)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_clusters(clusters_random, centroids_random,
              "K-Means with Random Initialization")

plot_clusters(clusters_pp, centroids_pp,
              "K-Means with K-Means++ Initialization")

# ------------------------------------------------------------
# End of File
# ------------------------------------------------------------
