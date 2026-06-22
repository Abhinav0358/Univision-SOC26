"""
Tree-Based Models and Ensembles

Covers tree splitting criteria (Entropy), ensemble methodologies (Bagging vs. Boosting),
feature importance, and decision boundary analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)

# Splitting Criteria
# Decision trees evaluate splits by measuring the reduction in impurity:
# - Entropy: Quantifies informational disorder or surprise.
# - Gini Impurity: Measures the probability of misclassifying a randomly chosen element.


def calculate_entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # Entropy formula: -sum(p * log2(p))
    return -np.sum(probs * np.log2(probs + 1e-9))


# Ensemble Strategies
# Single decision trees are prone to high variance (overfitting).
# - Bagging (e.g., Random Forest): Trains independent trees on bootstrapped data subsets
#   and random feature subsets. Predictions are averaged to reduce variance.
# - Boosting (e.g., GBM, XGBoost): Fits trees sequentially, where each new tree is
#   trained to minimize the residual errors of the previous sequence. Reduces bias.

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

# Fit models
tree = DecisionTreeClassifier(max_depth=None, random_state=42).fit(X, y)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
gbm = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42
).fit(X, y)

# Feature Importance
# Computed as the average reduction in node impurity contributed by a feature.
importances = rf.feature_importances_
print("--- Feature Importances ---")
print(f"Feature 1: {importances[0]:.4f}, Feature 2: {importances[1]:.4f}")


# Decision Boundary Visualization
def plot_decision_boundary(model, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm", alpha=0.5)
    ax.set_title(title)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_decision_boundary(tree, X, y, "Decision Tree (Overfitted/Jagged)", axes[0])
plot_decision_boundary(rf, X, y, "Random Forest (Smoothed)", axes[1])
plot_decision_boundary(gbm, X, y, "Gradient Boosting (Balanced)", axes[2])

plt.savefig("lab07_results.png")
print("\n[SUCCESS] Tree ensemble evaluation completed. Plots saved to 'lab07_results.png'.")
