"""
Lab 06: Classification and Metrics - Deep Dive Tutorial
======================================================
Syllabus Topics: Sigmoid/Softmax, Logistic Regression, Confusion Matrix, Precision/Recall/F1, ROC/AUC.

This script explains how we predict categories and how we measure success.
In classification, Accuracy is often a LIE. We need better metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

# Set seed
np.random.seed(42)

# =============================================================================
# 1. THE ACTIVATION: SIGMOID & SOFTMAX
# =============================================================================
# - SIGMOID: Takes any number and squashes it between 0 and 1.
#   Used for BINARY classification. 0.7 means "70% chance of being class 1".
# - SOFTMAX: Used for MULTI-CLASS. It ensures all probabilities sum up to 100%.


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# =============================================================================
# 2. THE MODEL: LOGISTIC REGRESSION
# =============================================================================
# Despite the name, it is a CLASSIFIER.
# It fits a linear line (Decision Boundary) but applies Sigmoid to the output.

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],  # IMBALANCED
    random_state=42,
)
# Why weight 0.9/0.1? In real life (Fraud detection), one class is much rarer.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression().fit(X_train, y_train)
y_probs = model.predict_proba(X_test)[:, 1]  # The probability of being Class 1
y_pred = (y_probs > 0.5).astype(int)  # Applying a 0.5 threshold

# =============================================================================
# 3. EVALUATION METRICS (The Truth)
# =============================================================================
# If 90% of your data is "Normal", a model that ALWAYS says "Normal" is 90% accurate.
# But it is USELESS for finding "Fraud". We need:


def detailed_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # PRECISION: Of all we predicted as positive, how many were ACTUALLY positive?
    # "Don't cry wolf." (Focus on reducing False Positives)
    precision = tp / (tp + fp)

    # RECALL (Sensitivity): Of all actual positives, how many did we catch?
    # "Don't miss a thief." (Focus on reducing False Negatives)
    recall = tp / (tp + fn)

    # F1-SCORE: The harmonic mean of Precision and Recall. A balanced score.
    f1 = 2 * (precision * recall) / (precision + recall)

    # SPECIFICITY: Of all actual negatives, how many did we identify correctly?
    specificity = tn / (tn + fp)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
    }


print("--- Metrics at 0.5 Threshold ---")
print(detailed_metrics(y_test, y_pred))

# =============================================================================
# 4. ROC CURVE AND AUC (Threshold Analysis)
# =============================================================================
# What if we change the threshold from 0.5 to 0.1?
# We will catch MORE positives (Recall goes up) but have more false alarms (Precision goes down).
# The ROC CURVE plots Recall (True Pos Rate) vs 1-Specificity (False Pos Rate) for ALL thresholds.
# AUC (Area Under Curve) tells us the overall quality. 1.0 is perfect. 0.5 is a coin flip.

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
plt.figure(figsize=(15, 5))

# Plot 1: Sigmoid Visualization
plt.subplot(1, 3, 1)
z = np.linspace(-7, 7, 100)
plt.plot(z, sigmoid(z), color="blue")
plt.axvline(0, color="black", linestyle="--")
plt.axhline(0.5, color="red", linestyle="--")
plt.title("1. Sigmoid: Number -> Probability")

# Plot 2: Decision Boundary
plt.subplot(1, 3, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", alpha=0.3)
# Drawing the line where probability is 0.5
b = model.intercept_[0]
w1, w2 = model.coef_[0]
x1 = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 10)
x2 = -(b + w1 * x1) / w2
plt.plot(x1, x2, color="black", lw=2)
plt.title("2. Decision Boundary (Linear)")

# Plot 3: ROC Curve
plt.subplot(1, 3, 3)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("3. ROC Curve: Quality of Model")
plt.legend()

plt.tight_layout()
plt.savefig("lab06_results.png")
print("\n[SUCCESS] Lab 06 Complete. Check 'lab06_results.png'.")
