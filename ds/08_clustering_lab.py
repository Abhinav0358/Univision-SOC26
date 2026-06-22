"""
Unsupervised Learning: Clustering Algorithms

Implements K-Means clustering, cluster selection via the Elbow Method, 
silhouette score validation, and Hierarchical (Agglomerative) clustering with Dendrograms.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

np.random.seed(42)

# K-Means Clustering
# Iterative clustering algorithm that partitions data into K distinct clusters:
# 1. Initialize K cluster centroids randomly.
# 2. Assign each observation to the nearest centroid.
# 3. Recompute centroids as the mean of their assigned observations.
# 4. Repeat until convergence.

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Elbow Method (Hyperparameter Selection)
# Identifies the optimal cluster count by plotting Inertia (Within-Cluster Sum of Squares)
# against K. The optimal K is selected at the "elbow" point where the rate of decrease slows down.

inertias = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(kmeans.inertia_)

# Silhouette Coefficient
# Evaluates clustering quality by measuring how close each point is to its own cluster
# relative to neighboring clusters. Scores range from -1 (poor mapping) to 1 (highly separated).

best_kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
score = silhouette_score(X, best_kmeans.labels_)
print("--- Clustering Quality ---")
print(f"Silhouette Score (k=4): {score:.4f}")

# Hierarchical Clustering
# - Agglomerative: A bottom-up approach where each observation starts as a single cluster
#   and merges sequentially with the nearest cluster based on a linkage metric (e.g., Ward linkage).
# - Divisive: A top-down approach that starts with one all-encompassing cluster and splits iteratively.

linked = linkage(X, method="ward")

# Distance Metrics:
# - Euclidean: Standard straight-line distance between two points in Euclidean space.
# - Manhattan: Distance computed along axes at right angles (grid-based distance).
# - Cosine: Measures the angular similarity between vectors, ignoring magnitude variations.

# Visualization
plt.figure(figsize=(18, 5))

# Elbow Plot
plt.subplot(1, 3, 1)
plt.plot(K_range, inertias, "bx-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

# Fitted Clusters
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=best_kmeans.labels_, cmap="viridis", alpha=0.6)
plt.scatter(
    best_kmeans.cluster_centers_[:, 0],
    best_kmeans.cluster_centers_[:, 1],
    s=200,
    c="red",
    marker="X",
    label="Centroids",
)
plt.title("K-Means Clusters (k=4)")
plt.legend()

# Dendrogram Representation
plt.subplot(1, 3, 3)
dendrogram(linked, truncate_mode="lastp", p=12)
plt.title("Hierarchical Dendrogram")

plt.tight_layout()
plt.savefig("lab08_results.png")
print("\n[SUCCESS] Clustering lab completed. Plots saved to 'lab08_results.png'.")
