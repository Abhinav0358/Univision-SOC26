"""
Lab 02: Encoding and Dimensionality Reduction - Deep Dive Tutorial
==================================================================
Syllabus Topics: Categorical Encoding (Target, Hashing), Curse of Dimensionality, PCA, t-SNE.

This script explains how to handle non-numeric data and manage high-dimensional complexity.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Set seed
np.random.seed(42)

# =============================================================================
# 1. CATEGORICAL FEATURE ENCODING (Turning text into numbers)
# =============================================================================
# Models only understand numbers. How we convert text matters.

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

# A. LABEL ENCODING (for Ordinal data):
# Assigns 0, 1, 2... based on alphabetical order.
# Problem: It implies "Tokyo (7) > Berlin (1)", which is false for cities.
le = LabelEncoder()
df["size_encoded"] = le.fit_transform(df["size"])

# B. ONE-HOT ENCODING (for Nominal data):
# Creates a new binary column for every category.
# 1 if City is New York, else 0.
# Problem: If you have 1000 cities, you get 1000 new columns! (Sparsity)
df_ohe = pd.get_dummies(df["city"], prefix="city")

# C. TARGET (MEAN) ENCODING:
# Replaces the category with the average target value for that category.
# e.g., If New York has a 70% 'Target=1' rate, New York becomes 0.7.
# Powerful but prone to OVERFITTING (Leakage).
city_means = df.groupby("city")["target"].mean()
df["city_target_enc"] = df["city"].map(city_means)

# D. HASH ENCODING:
# Uses a hashing function to map categories to a fixed number of columns (e.g., 4).
# Great for high-cardinality (1000s of categories) and memory efficiency.
from sklearn.feature_extraction import FeatureHasher

hasher = FeatureHasher(n_features=4, input_type="string")
hashed_features = hasher.transform(df[["city"]].values).toarray()

# =============================================================================
# 2. CURSE OF DIMENSIONALITY (Why more features isn't always better)
# =============================================================================
# As you add features (dimensions), the "volume" of your space grows exponentially.
# Your data points stay the same, so they become "lonely" (sparse).
# In high dimensions, every point is far from every other point.
# Similarity (Euclidean distance) breaks down.
print("\n--- The Curse ---")
print(
    "Think of it like finding a needle in a line (1D), then a square (2D), then a cube (3D)."
)
print("It gets exponentially harder as dimensions grow.")

# =============================================================================
# 3. PRINCIPAL COMPONENT ANALYSIS - PCA (Linear Reduction)
# =============================================================================
# PCA finds new "directions" (Principal Components) that capture the MOST variance.
# PC1 is the direction of max spread. PC2 is the second max, etc.
# It's a LINEAR transformation (Rotation + Projection).

digits = load_digits()  # 64-dimensional data (8x8 images of numbers)
X, y = digits.data, digits.target

pca = PCA(n_components=0.95)  # "Keep enough components to explain 95% of the data"
X_pca = pca.fit_transform(X)

print(f"\nOriginal dimensions: {X.shape[1]}")
print(f"PCA reduced dimensions: {X_pca.shape[1]}")

# =============================================================================
# 4. t-SNE (Non-Linear Visualization)
# =============================================================================
# PCA often fails to separate complex clusters.
# t-SNE (t-Distributed Stochastic Neighbor Embedding) is "magical" for visualization.
# It tries to keep points that were close in high-D close in 2D.
# It is NON-LINEAR and STOCHASTIC (results change slightly each run).
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.7)
plt.colorbar(label="Digit Class")
plt.title("PCA: Linear Projection (Some overlap)")

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", alpha=0.7)
plt.colorbar(label="Digit Class")
plt.title("t-SNE: Non-Linear (Clear clusters!)")

plt.tight_layout()
plt.savefig("lab02_results.png")
print("\n[SUCCESS] Lab 02 Complete. Check 'lab02_results.png'.")
