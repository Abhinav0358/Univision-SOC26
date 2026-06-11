"""
Lab 03: Statistical Tests for Model Validity - Deep Dive Tutorial
================================================================
Syllabus Topics: Normality (Q-Q, Jarque-Bera), Multicollinearity (VIF), Autocorrelation (Durbin-Watson).

This script is about "Statistical Hygiene".
Before you trust a model, you must check if its assumptions are met.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

# Set seed
np.random.seed(42)

# =============================================================================
# 1. SYNTHETIC DATA GENERATION (With deliberate "flaws")
# =============================================================================
n_samples = 200
X1 = np.random.normal(0, 1, n_samples)
# X2 is almost identical to X1. This will cause MULTICOLLINEARITY.
X2 = X1 + np.random.normal(0, 0.01, n_samples)
X3 = np.random.normal(0, 1, n_samples)

# We use Exponential noise instead of Normal noise. This will fail NORMALITY checks.
noise = np.random.exponential(scale=1, size=n_samples) - 1
y = 2 * X1 + 0.5 * X3 + noise

df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "y": y})

# =============================================================================
# 2. MULTICOLLINEARITY (The VIF Test)
# =============================================================================
# VIF (Variance Inflation Factor):
# Tells you if your independent variables (X1, X2, X3) are "predicting each other".
# If they are, the model gets confused about which one is actually causing 'y'.
# - VIF = 1: Not correlated.
# - VIF > 5: Moderate correlation.
# - VIF > 10: SEVERE correlation (Remove one of the variables!).


def calculate_vif(X_df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_df.columns
    # Formula: VIF = 1 / (1 - R^2) for each variable against the others.
    vif_data["VIF"] = [
        variance_inflation_factor(X_df.values, i) for i in range(len(X_df.columns))
    ]
    return vif_data


print("--- Multicollinearity (VIF) ---")
X = df[["X1", "X2", "X3"]]
X_with_const = sm.add_constant(X)  # Statsmodels needs a constant (intercept) column
print(calculate_vif(X_with_const))

# =============================================================================
# 3. MODEL FITTING & RESIDUALS
# =============================================================================
# Residuals = Actual Value (y) - Predicted Value (y_hat).
# Linear Regression assumes residuals are "Normal" and "Independent".
model = sm.OLS(y, X_with_const).fit()
residuals = model.resid

# =============================================================================
# 4. NORMALITY OF RESIDUALS (Are your errors random?)
# =============================================================================
# - Q-Q PLOT (Quantile-Quantile):
#   Plots your residuals against a perfect normal distribution.
#   If the points follow the straight red line, your residuals are Normal.
# - JARQUE-BERA TEST:
#   A statistical test for Normality based on Skewness and Kurtosis.
#   - Null Hypothesis (H0): Data is Normal.
#   - If p-value < 0.05, we REJECT H0 (Data is NOT Normal).

jb_test = stats.jarque_bera(residuals)
print(f"\nJarque-Bera p-value: {jb_test[1]:.4f}")

# =============================================================================
# 5. AUTOCORRELATION (The Durbin-Watson Test)
# =============================================================================
# Checks if your errors are correlated with each other (common in time-series).
# e.g., If the error today depends on the error yesterday, your model is missing a pattern.
# - DW = 2.0: Perfectly independent errors (Good!).
# - DW < 1.5: Positive autocorrelation (Errors "clump" together).
# - DW > 2.5: Negative autocorrelation.

dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson Stat: {dw_stat:.4f}")

# =============================================================================
# 6. VISUALIZATION
# =============================================================================
plt.figure(figsize=(15, 5))

# 1. Residuals vs Fitted: Points should be a random cloud (no shape/funnel).
plt.subplot(1, 3, 1)
sns.scatterplot(x=model.fittedvalues, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.title("1. Residuals vs Fitted (Cloud?)")

# 2. Q-Q Plot: Points should follow the red line.
plt.subplot(1, 3, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("2. Q-Q Plot (On the line?)")

# 3. Histogram: Should look like a Bell Curve.
plt.subplot(1, 3, 3)
sns.histplot(residuals, kde=True)
plt.title("3. Histogram (Bell Curve?)")

plt.tight_layout()
plt.savefig("lab03_results.png")
print("\n[SUCCESS] Lab 03 Complete. Check 'lab03_results.png'.")
