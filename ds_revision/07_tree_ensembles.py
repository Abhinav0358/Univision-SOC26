"""
Lab 07: Tree-Based Models and Ensembles - Deep Dive Tutorial
===========================================================
Syllabus Topics: Gini Impurity, Entropy, Bagging (Random Forest), Boosting (XGBoost/GBM).

This script explores the logic of "splitting" data and the power of "crowds" (Ensembles).
Trees are great because they are non-linear and easy to interpret.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Set seed
np.random.seed(42)

# =============================================================================
# 1. SPLIT CRITERIA (How a Tree decides where to cut)
# =============================================================================
# Trees want to create "pure" leaf nodes (only one class in the node).
# - ENTROPY: Measures "disorder" or "surprise".
#   Entropy = 0 means perfect purity. 1 means a 50/50 messy split.
# - GINI IMPURITY: Similar to Entropy but slightly faster to calculate.
#   It's the probability of incorrectly classifying a random element.


def calculate_entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # Entropy = -sum(p * log2(p))
    return -np.sum(probs * np.log2(probs + 1e-9))


# =============================================================================
# 2. ENSEMBLE METHODS (Strength in Numbers)
# =============================================================================
# A single tree is prone to OVERFITTING (High Variance). Ensembles fix this.

# A. BAGGING (Bootstrap Aggregating) -> e.g., RANDOM FOREST:
# - We train 100 trees on RANDOM subsets of data and RANDOM subsets of features.
# - They all vote. Averaging their votes reduces VARIANCE (errors cancel out).

# B. BOOSTING -> e.g., GBM, XGBoost, LightGBM:
# - Trees are trained SEQUENTIALLY.
# - Tree 2 tries to fix the mistakes of Tree 1.
# - This reduces BIAS (makes the model more accurate).

# =============================================================================
# 3. IMPLEMENTATION
# =============================================================================
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)  # Non-linear "Moon" shape

# 1. Single Tree
tree = DecisionTreeClassifier(max_depth=None, random_state=42).fit(X, y)

# 2. Random Forest (Bagging)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

# 3. Gradient Boosting (Boosting)
gbm = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42
).fit(X, y)

# =============================================================================
# 4. FEATURE IMPORTANCE (The "Why")
# =============================================================================
# Trees tell us which features are most useful.
# Importance = How much did this feature reduce Gini/Entropy on average?
importances = rf.feature_importances_
print(f"--- Feature Importances ---")
print(f"Feature 1: {importances[0]:.4f}, Feature 2: {importances[1]:.4f}")


# =============================================================================
# 5. VISUALIZATION (Decision Boundaries)
# =============================================================================
def plot_decision_boundary(model, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm", alpha=0.5)
    ax.set_title(title)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_decision_boundary(tree, X, y, "1. Decision Tree (Jagged/Overfit)", axes[0])
plot_decision_boundary(rf, X, y, "2. Random Forest (Smoother/Stable)", axes[1])
plot_decision_boundary(gbm, X, y, "3. Gradient Boosting (Precise)", axes[2])

plt.savefig("lab07_results.png")
print("\n[SUCCESS] Lab 07 Complete. Check 'lab07_results.png'.")
