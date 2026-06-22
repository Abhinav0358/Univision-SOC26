"""
Feature Encoding and Dimensionality Reduction

Covers categorical encoding techniques (Label, One-Hot, Target, Hash encoding) 
and dimensionality reduction methods (PCA, t-SNE) for high-dimensional data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher

np.random.seed(42)

# 1. Categorical Feature Encoding
n_samples = 500
df = pd.DataFrame(
    {
        "city": np.random.choice(
            [
                "New York",
                "London",
                "Paris",
                "Tokyo",
                "Mumbai",
                "Sydney",
                "Berlin",
                "Dubai",
            ],
            n_samples,
        ),
        "size": np.random.choice(
            ["Small", "Medium", "Large", "Extra Large"], n_samples
        ),
        "target": np.random.randint(0, 2, n_samples),
    }
)

# Label Encoding: Map ordinal categories to ranked integers
le = LabelEncoder()
df["size_encoded"] = le.fit_transform(df["size"])

# One-Hot Encoding: Map nominal categories to binary columns
# Generates a dummy column per category, which can cause high sparsity if cardinality is high
df_ohe = pd.get_dummies(df["city"], prefix="city")

# Target (Mean) Encoding: Replace category with the mean of the target variable for that category
# Fast and effective, but prone to target leakage and overfitting
city_means = df.groupby("city")["target"].mean()
df["city_target_enc"] = df["city"].map(city_means)

# Hash Encoding: Uses a hash function to map categories to a fixed number of columns
# Useful for high-cardinality features to limit the output dimension
hasher = FeatureHasher(n_features=4, input_type="string")
hashed_features = hasher.transform(df[["city"]].values).toarray()

# 2. Curse of Dimensionality
# High-dimensional space grows exponentially, making data points very sparse.
# Distance metrics like Euclidean distance start to fail because points become equidistant.
print("\n--- Dimensionality Concepts ---")
print("High-dimensional data leads to sparsity, making distance metrics less informative.")

# 3. Principal Component Analysis (PCA)
# Linear reduction technique that projects data onto directions of maximum variance.
digits = load_digits()  # 64-dimensional pixel data
X, y = digits.data, digits.target

pca = PCA(n_components=0.95)  # Keep enough components to capture 95% of variance
X_pca = pca.fit_transform(X)

print(f"\nOriginal feature count: {X.shape[1]}")
print(f"Reduced feature count via PCA (95% variance): {X_pca.shape[1]}")

# 4. t-Distributed Stochastic Neighbor Embedding (t-SNE)
# Non-linear probabilistic technique tailored for visual cluster analysis.
# Maintains local relationships, making neighbors in high-D remain neighbors in 2D.
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# 5. Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.7)
plt.colorbar(label="Digit Class")
plt.title("PCA: Linear Projection")

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", alpha=0.7)
plt.colorbar(label="Digit Class")
plt.title("t-SNE: Non-Linear Clusters")

plt.tight_layout()
plt.savefig("lab02_results.png")
print("\n[SUCCESS] Encoding and reduction completed. Results saved to 'lab02_results.png'.")
