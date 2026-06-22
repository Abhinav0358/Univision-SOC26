"""
Classification Foundations and Evaluation Metrics

Implements classification using Logistic Regression and evaluates model success
using Confusion Matrix metrics (Precision, Recall, F1, Specificity) and ROC/AUC curves.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Activation Functions:
# - Sigmoid: Maps real-valued inputs to a range between 0 and 1. Typically used for binary probability output.
# - Softmax: Generalizes sigmoid to multi-class settings, outputting a probability distribution that sums to 1.


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Model Fitting: Logistic Regression Classifier
# Generates a linear decision boundary by fitting weights to log-odds.

# Simulating imbalanced dataset to mirror real-world anomaly/fraud detection problems
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression().fit(X_train, y_train)
y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities of positive class
y_pred = (y_probs > 0.5).astype(int)

# Evaluation Metrics
# Precision, Recall, Specificity, and F1 provide a more realistic diagnostic of
# class-imbalanced models compared to basic accuracy.


def detailed_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Precision: True Positives out of all predicted Positives. Reduces false positives.
    precision = tp / (tp + fp)

    # Recall: True Positives out of all actual Positives. Reduces false negatives.
    recall = tp / (tp + fn)

    # F1-Score: Harmonic mean of Precision and Recall.
    f1 = 2 * (precision * recall) / (precision + recall)

    # Specificity: True Negatives out of all actual Negatives.
    specificity = tn / (tn + fp)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
    }


print("--- Evaluation Metrics at 0.5 Decision Threshold ---")
print(detailed_metrics(y_test, y_pred))

# Threshold Analysis via ROC and AUC
# A Receiver Operating Characteristic (ROC) curve evaluates TPR (Recall) against FPR
# (1 - Specificity) across all potential decision thresholds.
# AUC (Area Under Curve) aggregates the diagnostic performance of the classifier.

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Visualization
plt.figure(figsize=(15, 5))

# Plot Sigmoid mapping
plt.subplot(1, 3, 1)
z = np.linspace(-7, 7, 100)
plt.plot(z, sigmoid(z), color="blue")
plt.axvline(0, color="black", linestyle="--")
plt.axhline(0.5, color="red", linestyle="--")
plt.title("Sigmoid Function")

# Plot decision boundary
plt.subplot(1, 3, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", alpha=0.3)
b = model.intercept_[0]
w1, w2 = model.coef_[0]
x1 = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 10)
x2 = -(b + w1 * x1) / w2
plt.plot(x1, x2, color="black", lw=2)
plt.title("Linear Decision Boundary")

# Plot ROC curve
plt.subplot(1, 3, 3)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig("lab06_results.png")
print("\n[SUCCESS] Classification evaluation completed. Plots saved to 'lab06_results.png'.")
