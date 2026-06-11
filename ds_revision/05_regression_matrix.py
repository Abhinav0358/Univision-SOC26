"""
Lab 05: Regression Matrix and Model Selection - Deep Dive Tutorial
==================================================================
Syllabus Topics: OLS Matrix Representation, Polynomial Regression, AIC/BIC.

This script explains how models are solved mathematically and how to choose
the right level of complexity (Model Selection).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Set seed
np.random.seed(42)

# =============================================================================
# 1. THE MATH: OLS MATRIX REPRESENTATION (The Normal Equation)
# =============================================================================
# In simple code, we use .fit(). In math, Linear Regression is a Matrix problem.
# If Y = X * theta, then:
# theta = (X^T * X)^-1 * X^T * Y
# This is called the "Normal Equation". It solves for the optimal weights in ONE shot.


def solve_ols_matrix(X, y):
    # Add a column of 1s for the Intercept (Bias)
    X_b = np.c_[np.ones((len(X), 1)), X]
    # .T is Transpose, .dot is Matrix Multiplication, linalg.inv is Matrix Inverse
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best


# Toy Data
X_simple = 2 * np.random.rand(10, 1)
y_simple = 4 + 3 * X_simple + np.random.randn(10, 1)

weights = solve_ols_matrix(X_simple, y_simple)
print(f"--- OLS Matrix Solution ---")
print(f"Intercept: {weights[0][0]:.4f}, Slope: {weights[1][0]:.4f}")

# =============================================================================
# 2. POLYNOMIAL REGRESSION (Modeling Curves)
# =============================================================================
# Sometimes a straight line (Linear) is too simple (High Bias).
# We add powers of X (X^2, X^3...) to allow the model to bend.
# Danger: High degree polynomials can "memorize" the noise (Overfitting/High Variance).

n_samples = 50
X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, n_samples)

# =============================================================================
# 3. MODEL SELECTION CRITERIA (AIC and BIC)
# =============================================================================
# How do we pick the best polynomial degree?
# We look for a balance between "Good Fit" and "Simplicity".
# - AIC (Akaike Information Criterion): Penalizes adding parameters.
# - BIC (Bayesian Information Criterion): Penalizes parameters MORE than AIC.
# GOAL: Minimize AIC or BIC. The lowest value is the "Best" model.


def calculate_selection_criteria(n, rss, k):
    """
    n: number of samples
    rss: Residual Sum of Squares (Total Error)
    k: number of parameters (complexity)
    """
    # Formulas adapted for OLS
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return aic, bic


degrees = [1, 2, 3, 10, 20]  # 1 is too simple, 20 is too complex
plt.figure(figsize=(12, 7))
plt.scatter(X, y, color="black", label="Data")

results = []
for d in degrees:
    poly_features = PolynomialFeatures(degree=d, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    reg = LinearRegression().fit(X_poly, y)
    y_pred = reg.predict(X_poly)

    rss = np.sum((y - y_pred) ** 2)
    k = X_poly.shape[1] + 1  # +1 for intercept

    aic, bic = calculate_selection_criteria(n_samples, rss, k)
    results.append({"Degree": d, "AIC": aic, "BIC": bic})

    # Plotting the curve
    X_new = np.linspace(0, 5, 100).reshape(-1, 1)
    y_new = reg.predict(poly_features.transform(X_new))
    plt.plot(X_new, y_new, label=f"Deg {d} (AIC:{aic:.0f})")

# =============================================================================
# 4. RESULTS ANALYSIS
# =============================================================================
res_df = pd.DataFrame(results)
print("\n--- Model Comparison ---")
print(res_df)

print(
    "\nInsight: Notice how Deg 20 has the lowest RSS (best fit) but a HUGE AIC/BIC penalty."
)
print("The best model is usually a 'sweet spot' like Degree 3 or 4.")

plt.ylim(-1.5, 2.5)
plt.title("Polynomial Regression: The Search for the 'Best' Degree")
plt.legend()
plt.savefig("lab05_results.png")
print("\n[SUCCESS] Lab 05 Complete. Check 'lab05_results.png'.")
