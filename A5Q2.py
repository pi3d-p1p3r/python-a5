import numpy as np
import matplotlib.pyplot as plt

# Input data: House sizes and prices
X = np.array([[2104, 5], [1600, 3], [2400, 4]])
y = np.array([399900, 329900, 369000])
m = len(y)  # Number of training examples

# Feature normalization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# Add intercept term (bias)
X_b = np.c_[np.ones(m), X_norm]

# Initialize parameters (weights)
theta = np.zeros(X_b.shape[1])

# Set hyperparameters
alpha = 0.01  # Learning rate
iterations = 400 # Number of iterations

def gradient_descent(X, y, theta, alpha, iterations):
    """
    Performs gradient descent to learn theta.

    Args:
        X: Input features (with intercept term).
        y: Target values.
        theta: Initial parameters.
        alpha: Learning rate.
        iterations: Number of iterations.

    Returns:
        The learned theta.
    """
    for _ in range(iterations):
        gradient = (1 / m) * (X.T @ (X @ theta - y))
        theta -= alpha * gradient
    return theta

# Run gradient descent to learn theta
theta = gradient_descent(X_b, y, theta, alpha, iterations)

# Print the learned parameters
print("Learned theta:", theta)