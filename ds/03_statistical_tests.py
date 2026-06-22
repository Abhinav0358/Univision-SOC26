"""
Statistical Diagnostic Tests

Covers multicollinearity (VIF), residual analysis, error normality checks 
(Q-Q plots, Jarque-Bera), and residual autocorrelation (Durbin-Watson).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

np.random.seed(42)

# 1. Synthetic Data Generation (Introducing multicollinearity and non-normal errors)
n_samples = 200
X1 = np.random.normal(0, 1, n_samples)
# X2 is highly collinear with X1 (engineered to trigger high VIF)
X2 = X1 + np.random.normal(0, 0.01, n_samples)
X3 = np.random.normal(0, 1, n_samples)

# Using exponential noise to violate the normal residuals assumption
noise = np.random.exponential(scale=1, size=n_samples) - 1
y = 2 * X1 + 0.5 * X3 + noise

df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "y": y})

# 2. Multicollinearity Assessment via VIF
# Variance Inflation Factor measures how much the variance of an estimated regression
# coefficient is increased due to collinearity with other predictors.
# - VIF = 1: No correlation.
# - VIF > 5: Moderate collinearity.
# - VIF > 10: Severe collinearity (typically requires dropping a redundant feature).


def calculate_vif(X_df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_df.columns
    # VIF is calculated as 1 / (1 - R^2) of the feature regressed against all other features
    vif_data["VIF"] = [
        variance_inflation_factor(X_df.values, i) for i in range(len(X_df.columns))
    ]
    return vif_data


print("--- Multicollinearity (VIF) ---")
X = df[["X1", "X2", "X3"]]
X_with_const = sm.add_constant(X)  # Constant required for statsmodels intercept
print(calculate_vif(X_with_const))

# 3. Model Fitting and Residual Extraction
# Residuals (y - y_hat) represent error terms. Linear regression assumes independent,
# normally distributed error terms.
model = sm.OLS(y, X_with_const).fit()
residuals = model.resid

# 4. Residual Normality Diagnostics
# - Q-Q Plot: Compares the quantiles of sample residuals against theoretical normal quantiles.
# - Jarque-Bera Test: Goodness-of-fit test checking if skewness and kurtosis match a normal distribution.
#   Null Hypothesis (H0): Residuals are normally distributed. Reject H0 if p-value < 0.05.
jb_test = stats.jarque_bera(residuals)
print(f"\nJarque-Bera p-value: {jb_test[1]:.4f}")

# 5. Autocorrelation Check via Durbin-Watson
# Detects presence of serial correlation in residuals (common in time-series data).
# - Value near 2.0 indicates no serial correlation.
# - Value close to 0 indicates positive autocorrelation.
# - Value close to 4 indicates negative autocorrelation.
dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson Stat: {dw_stat:.4f}")

# 6. Diagnostic Visualization
plt.figure(figsize=(15, 5))

# Residuals vs Fitted values should look like a random scatter (homoscedasticity)
plt.subplot(1, 3, 1)
sns.scatterplot(x=model.fittedvalues, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Fitted")

# Points should follow the 45-degree diagonal line
plt.subplot(1, 3, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot")

# Error distribution check
plt.subplot(1, 3, 3)
sns.histplot(residuals, kde=True)
plt.title("Residual Histogram")

plt.tight_layout()
plt.savefig("lab03_results.png")
print("\n[SUCCESS] Statistical checks completed. Diagnostic plots saved to 'lab03_results.png'.")
