"""
Data Preprocessing: Cleaning and Scaling

Covers levels of measurement, missing data imputation, outlier detection,
and feature scaling techniques.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

np.random.seed(42)

# 1. Levels of Measurement
# Understanding variable types to choose correct preprocessing methods:
# - Nominal: Categories with no inherent ordering (e.g., zip codes, colors).
# - Ordinal: Categories with a logical order, but non-uniform intervals (e.g., ratings).
# - Interval: Numerical values with uniform intervals but no absolute zero (e.g., Celsius).
# - Ratio: Numerical values with a true zero representing absence of quantity (e.g., income, weight).

# 2. Synthetic Data Generation
n_samples = 1000
data = pd.DataFrame(
    {
        # Using a log-normal distribution to represent typical skewed real-world income distributions
        "income": np.random.lognormal(mean=10, sigma=1, size=n_samples),
        "age": np.random.randint(18, 80, size=n_samples).astype(float),
        "experience": np.random.normal(loc=10, scale=5, size=n_samples),
    }
)

print("--- Sample Data ---")
print(data.head(10))

# 3. Missing Data Types
# Simulating different missingness mechanisms:
# - MCAR (Missing Completely at Random): Missingness is independent of any observed/unobserved data.
# - MAR (Missing at Random): Missingness depends on other observed features (e.g., missing experience correlates with age).
# - MNAR (Missing Not at Random): Missingness depends on the value of the missing variable itself.

# Simulate MAR: experience goes missing for some records where age > 60
mask = (data["age"] > 60) & (np.random.rand(n_samples) > 0.5)
data.loc[mask, "experience"] = np.nan

print("Dataset with missing values introduced:")
print(data.head(20))

print("--- Missing Data Summary ---")
print(data.isnull().sum())

# 4. Imputation
# Compare basic mean imputation against multi-variable imputation:
# - Mean/Median: Computationally cheap but distorts variance and correlation.
# - KNN Imputer: Uses Euclidean distance of neighboring records to estimate missing values.

data_simple = data.copy()
data_simple["exp_mean"] = data_simple["experience"].fillna(
    data_simple["experience"].mean()
)

imputer = KNNImputer(n_neighbors=5)
data_knn = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 5. Outlier Detection & Capping
# Identifying extreme values using the 1.5 * IQR rule.
# - IQR (Interquartile Range) measures the spread of the middle 50% of the dataset.

# Introduce artificial outliers
data.loc[:10, "income"] = data["income"].max() * 5


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (
        df[(df[column] < lower_bound) | (df[column] > upper_bound)],
        lower_bound,
        upper_bound,
    )


outliers, lb, ub = detect_outliers_iqr(data, "income")

# Winsorization: Cap outliers at the IQR boundary thresholds instead of dropping them
data_capped = data.copy()
data_capped["income"] = np.where(
    data_capped["income"] > ub,
    ub,
    np.where(data_capped["income"] < lb, lb, data_capped["income"]),
)

# 6. Feature Scaling & Transformations
# - Log transformation: Reduces skewness and handles large ranges of values.
# - Z-Score Standardization: Centers data (mean=0, variance=1), required for PCA and SVMs.
# - Min-Max Normalization: Rescales features strictly into the range [0, 1].

data["income_log"] = np.log1p(data["income"])

scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

data["income_std"] = scaler_std.fit_transform(data[["income"]])
data["income_minmax"] = scaler_minmax.fit_transform(data[["income"]])

# 7. Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(data["income"], kde=True)
plt.title("Original Skewed Income")

plt.subplot(2, 2, 2)
sns.histplot(data["income_log"], kde=True)
plt.title("Log-Transformed Income")

plt.subplot(2, 2, 3)
sns.kdeplot(data["income_std"], label="Standardized (Z-Score)")
sns.kdeplot(data["income_minmax"], label="Normalized (Min-Max)")
plt.title("Scaling Comparison")
plt.legend()

plt.subplot(2, 2, 4)
sns.boxplot(x=data_capped["income"])
plt.title("Income Boxplot (After Capping)")

plt.tight_layout()
plt.savefig("lab01_results.png")
print("\n[SUCCESS] Preprocessing completed. Results saved to 'lab01_results.png'.")
