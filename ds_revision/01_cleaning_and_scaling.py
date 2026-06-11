"""
Lab 01: Data Cleaning and Scaling - Deep Dive Tutorial
=====================================================
Syllabus Topics: Levels of Measurement, Missing Data, Outlier Detection, Scaling.

This script is your guide to the first step of any DS project: PREPROCESSING.
Data is almost always messy, biased, or incorrectly scaled.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set seed for reproducibility (makes sure random results are the same every time you run it)
np.random.seed(42)

# =============================================================================
# 1. LEVELS OF MEASUREMENT (The "Type" of your data)
# =============================================================================
# Before you code, you must know what your data is.
# - NOMINAL: Pure categories. No order. (e.g., Eye color, Zip code). You can't say "Red > Blue".
# - ORDINAL: Categories with a RANK. (e.g., "Poor", "Good", "Excellent"). Order matters, but the gap isn't fixed.
# - INTERVAL: Numbers where the gap is meaningful, but 0 is NOT "nothing". (e.g., Temperature in Celsius).
#             0°C doesn't mean "no heat".
# - RATIO: Numbers with a TRUE ZERO. (e.g., Income, Weight, Distance). $0 means "no money".
#   Most ML models love Ratio data.

# =============================================================================
# 2. SYNTHETIC DATA GENERATION
# =============================================================================
n_samples = 1000
data = pd.DataFrame(
    {
        # We use log-normal because real-world income is usually skewed (a few rich people)
        "income": np.random.lognormal(mean=10, sigma=1, size=n_samples),
        "age": np.random.randint(18, 80, size=n_samples).astype(float),
        "experience": np.random.normal(loc=10, scale=5, size=n_samples),
    }
)

# =============================================================================
# 3. MISSING DATA (The "How" and "Why" of missing values)
# =============================================================================
# - MCAR (Missing Completely At Random): The missingness is a random accident.
# - MAR (Missing At Random): The missingness depends on OTHER observed data.
#   (e.g., Older people (Age) are less likely to report their Experience).
# - MNAR (Missing Not At Random): The missingness depends on the value itself.
#   (e.g., People with VERY high income don't want to report it).

# Simulation of MAR: Experience is missing ONLY if age > 60
mask = (data["age"] > 60) & (np.random.rand(n_samples) > 0.5)
data.loc[mask, "experience"] = np.nan

print("--- Missing Data Summary ---")
print(data.isnull().sum())

# =============================================================================
# 4. IMPUTATION (Filling in the holes)
# =============================================================================
# - MEAN/MEDIAN Imputation: Quick but "blunt". It reduces variance (makes data look too average).
# - KNN (k-Nearest Neighbors) Imputation: Sophisticated. It looks at the most similar 5 people
#   (based on Age and Income) and uses their average Experience to fill the hole.

# Simple Imputation
data_simple = data.copy()
data_simple["exp_mean"] = data_simple["experience"].fillna(
    data_simple["experience"].mean()
)

# KNN Imputation
# KNNImputer calculates Euclidean distance between rows to find "neighbors".
imputer = KNNImputer(n_neighbors=5)
data_knn = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# =============================================================================
# 5. OUTLIER DETECTION (The 1.5 IQR Rule)
# =============================================================================
# Outliers can ruin models (especially Linear Regression).
# - IQR (Interquartile Range): The middle 50% of your data (Q3 - Q1).
# - 1.5 IQR Rule: Anything beyond [Q1 - 1.5*IQR] or [Q3 + 1.5*IQR] is an outlier.

# Injecting artificial outliers
data.loc[:10, "income"] = data["income"].max() * 5


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filtering the data for values outside these bounds
    return (
        df[(df[column] < lower_bound) | (df[column] > upper_bound)],
        lower_bound,
        upper_bound,
    )


outliers, lb, ub = detect_outliers_iqr(data, "income")

# - WINSORIZATION (Capping): Instead of deleting outliers (which loses data),
#   we "squash" them to the upper/lower bounds.
data_capped = data.copy()
data_capped["income"] = np.where(
    data_capped["income"] > ub,
    ub,
    np.where(data_capped["income"] < lb, lb, data_capped["income"]),
)

# =============================================================================
# 6. TRANSFORMATION & SCALING (Making data "comparable")
# =============================================================================
# - LOG TRANSFORMATION: Used to "un-skew" data. It compresses large values more than small ones.
# - STANDARDIZATION (Z-Score): Scales data to mean=0 and std=1.
#   Essential for algorithms like PCA or SVM that assume centered data.
# - NORMALIZATION (Min-Max): Scales data between 0 and 1.
#   Useful when you have clear bounds (like pixel values 0-255).

data["income_log"] = np.log1p(data["income"])  # log(1+x) to avoid log(0) errors

scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

data["income_std"] = scaler_std.fit_transform(data[["income"]])
data["income_minmax"] = scaler_minmax.fit_transform(data[["income"]])

# =============================================================================
# 7. VISUALIZATION (Verify with your eyes!)
# =============================================================================
plt.figure(figsize=(15, 10))

# 1. Shows how the original data was skewed (pushed to the left)
plt.subplot(2, 2, 1)
sns.histplot(data["income"], kde=True)
plt.title("1. Original Income (Highly Skewed)")

# 2. Shows how Log transformation makes the distribution look "Normal" (Bell-shaped)
plt.subplot(2, 2, 2)
sns.histplot(data["income_log"], kde=True)
plt.title("2. Log Transformed (Bell Curve)")

# 3. Shows how Z-score and Min-Max look on the same scale
plt.subplot(2, 2, 3)
sns.kdeplot(data["income_std"], label="Standardized (Center=0)")
sns.kdeplot(data["income_minmax"], label="Normalized (Bound [0,1])")
plt.title("3. Scaling Comparison")
plt.legend()

# 4. Shows the Boxplot after we 'capped' the outliers. No points should be beyond the whiskers.
plt.subplot(2, 2, 4)
sns.boxplot(x=data_capped["income"])
plt.title("4. Income after Capping Outliers")

plt.tight_layout()
plt.savefig("lab01_results.png")
print("\n[SUCCESS] Lab 01 Complete. Check 'lab01_results.png' for visual proof.")
