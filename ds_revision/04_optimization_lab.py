"""
Lab 04: Optimization Lab - Deep Dive Tutorial
=============================================
Syllabus Topics: Loss vs. Cost Functions, Gradient Descent, Newton-Raphson.

This script implements the "engine" that powers almost all ML: OPTIMIZATION.
How does a model "learn" parameters? By minimizing error.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set seed
np.random.seed(42)

# =============================================================================
# 1. CORE CONCEPTS
# =============================================================================
# - LOSS FUNCTION (L):
#   The error for a SINGLE prediction.
#   e.g., Squared Error: (y - y_hat)^2
# - COST FUNCTION (J):
#   The average of all losses across the WHOLE dataset.
#   e.g., Mean Squared Error (MSE): (1/n) * sum(Losses)
# - GRADIENT DESCENT:
#   An iterative algorithm to find the minimum of the Cost Function.
#   It's like walking down a mountain in the fog by feeling the slope (gradient).

# =============================================================================
# 2. DATA GENERATION
# =============================================================================
# Ground truth: y = 2x + 1
X = 2 * np.random.rand(100, 1)
y = 1 + 2 * X + np.random.randn(100, 1) * 0.2

# =============================================================================
# 3. GRADIENT DESCENT FROM SCRATCH
# =============================================================================


def compute_cost(X, y, theta):
    """Calculates the MSE for current parameters (theta)."""
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    """
    Finds optimal theta by repeatedly taking steps 'downhill'.
    - learning_rate: The size of the step. Too big? You jump over the hole. Too small? You take forever.
    """
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        prediction = np.dot(X, theta)
        # THE GRADIENT: The partial derivative of the Cost function with respect to theta.
        # It tells us which way is "up". We subtract it to go "down".
        gradient = (1 / m) * (X.T.dot(prediction - y))
        theta = theta - learning_rate * gradient

        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history


# X_b adds a column of 1s to handle the Intercept (b in y=mx+b)
X_b = np.c_[np.ones((100, 1)), X]
theta_init = np.random.randn(2, 1)

# Test different Learning Rates
lrs = [0.01, 0.1, 0.5]
results = {lr: gradient_descent(X_b, y, theta_init.copy(), lr, 50)[1] for lr in lrs}

# =============================================================================
# 4. NEWTON-RAPHSON METHOD (Second-Order Optimization)
# =============================================================================
# Newton's method uses both the Gradient (slope) and the Hessian (curvature).
# Advantage: It converges MUCH faster (often in one step for linear models).
# Disadvantage: Calculating the Hessian is computationally expensive for many variables.
# Formula: theta = (X^T * X)^-1 * X^T * y (The Normal Equation is a special case)


def newton_raphson(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


theta_newton = newton_raphson(X_b, y)

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
plt.figure(figsize=(12, 5))

# Plot 1: How fast did we find the minimum?
plt.subplot(1, 2, 1)
for lr, history in results.items():
    plt.plot(range(len(history)), history, label=f"LR={lr}")
plt.title("1. Cost vs. Iterations (Learning Speed)")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.legend()

# Plot 2: The result
plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label="Data Points")
x_plot = np.array([[0], [2]])
x_plot_b = np.c_[np.ones((2, 1)), x_plot]
y_plot = x_plot_b.dot(theta_newton)
plt.plot(x_plot, y_plot, color="red", label="Optimal Fit (Newton)")
plt.title("2. Final Linear Model")
plt.legend()

plt.tight_layout()
plt.savefig("lab04_results.png")
print("\n[SUCCESS] Lab 04 Complete. Check 'lab04_results.png'.")
