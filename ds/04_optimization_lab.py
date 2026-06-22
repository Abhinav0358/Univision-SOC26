"""
Optimization Foundations

Implements Loss vs. Cost functions, Gradient Descent optimization, 
and the Newton-Raphson method (Normal Equation) for linear parameter estimation.
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Optimization Terminology:
# - Loss Function: The error calculation for a single training example.
# - Cost Function (Objective Function): The average loss across the entire dataset (e.g., MSE).
# - Gradient Descent: An iterative optimization algorithm that updates parameters by moving
#   in the opposite direction of the gradient (steepest descent).

# Generate synthetic linear dataset: y = 2x + 1 + noise
X = 2 * np.random.rand(100, 1)
y = 1 + 2 * X + np.random.randn(100, 1) * 0.2


def compute_cost(X, y, theta):
    """Calculates Mean Squared Error (MSE) for model parameters theta."""
    m = len(y)
    predictions = X.dot(theta)
    return (1 / (2 * m)) * np.sum(np.square(predictions - y))


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    """Updates parameters iteratively to minimize the cost function."""
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        prediction = np.dot(X, theta)
        # Compute partial derivatives of the cost function relative to each parameter
        gradient = (1 / m) * (X.T.dot(prediction - y))
        theta = theta - learning_rate * gradient
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history


# Prepare features with a bias column (column of ones for the intercept)
X_b = np.c_[np.ones((100, 1)), X]
theta_init = np.random.randn(2, 1)

# Evaluate parameter convergence across multiple learning rates
lrs = [0.01, 0.1, 0.5]
results = {lr: gradient_descent(X_b, y, theta_init.copy(), lr, 50)[1] for lr in lrs}

# Newton-Raphson Method
# Second-order optimization technique utilizing curvature (Hessian matrix) to locate minima.
# For linear models, the closed-form solution is the Normal Equation.
# Pros: Rapid convergence (typically single-step).
# Cons: Hessian inversion scales poorly with many features.
def newton_raphson(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


theta_newton = newton_raphson(X_b, y)

# Visualization
plt.figure(figsize=(12, 5))

# Convergence plot
plt.subplot(1, 2, 1)
for lr, history in results.items():
    plt.plot(range(len(history)), history, label=f"LR={lr}")
    plt.title("Convergence Rates by Learning Rate")
plt.xlabel("Iterations")
plt.ylabel("MSE Cost")
plt.legend()

# Fitted regression line
plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label="Observations")
x_plot = np.array([[0], [2]])
x_plot_b = np.c_[np.ones((2, 1)), x_plot]
y_plot = x_plot_b.dot(theta_newton)
plt.plot(x_plot, y_plot, color="red", label="Optimized Model")
plt.title("Fitted Linear Model")
plt.legend()

plt.tight_layout()
plt.savefig("lab04_results.png")
print("\n[SUCCESS] Optimization tasks completed. Plots saved to 'lab04_results.png'.")
