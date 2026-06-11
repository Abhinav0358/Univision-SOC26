"""
Lab 08: Unsupervised Learning and Clustering - Deep Dive Tutorial
================================================================
Syllabus Topics: K-Means, Silhouette Score, Hierarchical Clustering, Distance Metrics.

This script explains how to find groups in data when you DON'T have labels.
Useful for Customer Segmentation, Image Compression, and Anomaly Detection.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Set seed
np.random.seed(42)

# =============================================================================
# 1. K-MEANS CLUSTERING (The centroid approach)
# =============================================================================
# How it works:
# 1. Pick 'K' random points as "Centroids".
# 2. Assign every data point to its nearest Centroid.
# 3. Move Centroid to the center of its assigned points.
# 4. Repeat until centroids stop moving.

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# =============================================================================
# 2. ELBOW METHOD (Finding the optimal K)
# =============================================================================
# How do we know how many clusters (K) to use?
# - INERTIA (Within-Cluster Sum of Squares): Sum of squared distances from points to their centroid.
# - As K increases, Inertia always decreases.
# - We look for the "ELBOW" point where the decrease slows down.
#   That is our optimal K.

inertias = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(kmeans.inertia_)

# =============================================================================
# 3. SILHOUETTE SCORE (Quality check)
# =============================================================================
# Measures how similar a point is to its own cluster compared to other clusters.
# - Score = 1: Point is far from other clusters (Perfect).
# - Score = 0: Point is on the boundary between two clusters.
# - Score = -1: Point is likely in the WRONG cluster.
best_kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
score = silhouette_score(X, best_kmeans.labels_)
print(f"--- Clustering Quality ---")
print(f"Silhouette Score (k=4): {score:.4f}")

# =============================================================================
# 4. HIERARCHICAL CLUSTERING & DENDROGRAMS
# =============================================================================
# Instead of picking 'K', we build a "tree of clusters" (Dendrogram).
# - AGGLOMERATIVE: Bottom-up. Start with every point as a cluster and merge the closest ones.
# - DIVISIVE: Top-down. Start with one big cluster and split it.
# - LINKAGE: The rule for measuring distance between clusters (e.g., 'Ward' minimizes variance).

linked = linkage(X, method="ward")

# =============================================================================
# 5. DISTANCE METRICS (How to measure 'Nearness'?)
# =============================================================================
# - EUCLIDEAN: "As the crow flies" (Straight line).
# - MANHATTAN: "Taxicab distance" (Following a grid).
# - COSINE: Measures the ANGLE between vectors (Great for text, ignores magnitude).

# =============================================================================
# 6. VISUALIZATION
# =============================================================================
plt.figure(figsize=(18, 5))

# Plot 1: Elbow Method
plt.subplot(1, 3, 1)
plt.plot(K_range, inertias, "bx-")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("1. Elbow Method: Look for the 'Bend'")

# Plot 2: Final Clusters
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
plt.title("2. K-Means Final Clusters (k=4)")
plt.legend()

# Plot 3: Dendrogram
plt.subplot(1, 3, 3)
dendrogram(linked, truncate_mode="lastp", p=12)
plt.title("3. Dendrogram (Hierarchical Tree)")

plt.tight_layout()
plt.savefig("lab08_results.png")
print("\n[SUCCESS] Lab 08 Complete. Check 'lab08_results.png'.")
