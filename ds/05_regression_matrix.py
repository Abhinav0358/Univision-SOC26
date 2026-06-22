"""
Regression Matrix Representation and Model Selection

Implements Ordinary Least Squares (OLS) using matrix algebra, 
evaluates Polynomial Regression, and applies AIC/BIC metrics to balance 
model fit and parameter complexity.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)

# Ordinary Least Squares Matrix Representation
# Linear regression parameter estimation resolved via the Normal Equation:
# theta = (X^T * X)^-1 * X^T * y
# Computes the global minimum of the loss surface analytically.


def solve_ols_matrix(X, y):
    # Add a column of ones for intercept estimation
    X_b = np.c_[np.ones((len(X), 1)), X]
    # Matrix operations: Transpose, dot product, and matrix inverse
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best


# Generate simple toy dataset
X_simple = 2 * np.random.rand(10, 1)
y_simple = 4 + 3 * X_simple + np.random.randn(10, 1)

weights = solve_ols_matrix(X_simple, y_simple)
print("--- OLS Matrix Solution ---")
print(f"Intercept: {weights[0][0]:.4f}, Slope: {weights[1][0]:.4f}")

# Polynomial Regression
# Extends linear models to capture non-linear relationships by mapping input
# features into a higher-dimensional polynomial feature space.
# Avoid excessively high degrees to prevent overfitting of sample noise.

n_samples = 50
X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, n_samples)

# Model Selection Criteria: AIC & BIC
# Used to determine the optimal model complexity:
# - AIC (Akaike Information Criterion): Penalizes complexity based on parameter count.
# - BIC (Bayesian Information Criterion): Penalizes parameters more heavily than AIC as sample size grows.
# The model that minimizes these criteria represents the optimal trade-off.


def calculate_selection_criteria(n, rss, k):
    """
    n: Sample count
    rss: Residual Sum of Squares (unexplained variance)
    k: Number of model parameters (including intercept)
    """
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return aic, bic


degrees = [1, 2, 3, 10, 20]
plt.figure(figsize=(12, 7))
plt.scatter(X, y, color="black", label="Observations")

results = []
for d in degrees:
    poly_features = PolynomialFeatures(degree=d, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    reg = LinearRegression().fit(X_poly, y)
    y_pred = reg.predict(X_poly)

    rss = np.sum((y - y_pred) ** 2)
    k = X_poly.shape[1] + 1  # Parameter count + intercept

    aic, bic = calculate_selection_criteria(n_samples, rss, k)
    results.append({"Degree": d, "AIC": aic, "BIC": bic})

    # Plot fitted model regression curves
    X_new = np.linspace(0, 5, 100).reshape(-1, 1)
    y_new = reg.predict(poly_features.transform(X_new))
    plt.plot(X_new, y_new, label=f"Deg {d} (AIC:{aic:.0f})")

# Results evaluation
res_df = pd.DataFrame(results)
print("\n--- Model Comparison ---")
print(res_df)

print(
    "\nInsight: Although Degree 20 yields a lower RSS, it incurs a severe AIC/BIC complexity penalty."
)
print("The optimal balance is typically found around Degree 3 or 4.")

plt.ylim(-1.5, 2.5)
plt.title("Polynomial Fitting: Complexity vs. Model Fit")
plt.legend()
plt.savefig("lab05_results.png")
print("\n[SUCCESS] Regression matrix lab completed. Plots saved to 'lab05_results.png'.")
