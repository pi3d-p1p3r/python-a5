import numpy as np
import matplotlib.pyplot as plt

# ── 1. Load data from file ──────────────────────────────────────────
data = np.loadtxt("ex2data1.txt", delimiter=",")
X = data[:, 0]          # Population (x)
y = data[:, 1]          # Profit (y)
m = len(y)               # Number of training examples

# ── 2. Feature normalization ────────────────────────────────────────
X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()
X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

# ── 3. Setup ────────────────────────────────────────────────────────
X_b = np.c_[np.ones(m), X_norm]  # Add intercept term (bias column of 1s)
theta = np.zeros(2)               # Initialize theta to [0, 0]
alpha = 0.01                      # Learning rate
iterations = 1500                  # Number of gradient descent steps

# ── 4. Cost function ────────────────────────────────────────────────
def compute_cost(X, y, theta):
    errors = X @ theta - y
    return (1 / (2 * m)) * np.dot(errors, errors)

# ── 5. Gradient descent ─────────────────────────────────────────────
def gradient_descent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        gradient = (1 / m) * (X.T @ (X @ theta - y))
        theta -= alpha * gradient
    return theta

# ── 6. Train and print result ───────────────────────────────────────
theta = gradient_descent(X_b, y_norm, theta, alpha, iterations)

# Convert theta back to original scale
theta_original = np.array([
    y_mean + y_std * (theta[0] - theta[1] * X_mean / X_std),
    y_std * theta[1] / X_std
])

print("Learned theta:", theta_original)

# ── 7. Plot data + regression line ──────────────────────────────────
plt.scatter(X, y, marker='x', c='red', label='Training data')
plt.plot(X, theta_original[0] + theta_original[1] * X, 'b-', label='Linear regression')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear Regression with One Variable')
plt.legend()
plt.show()
