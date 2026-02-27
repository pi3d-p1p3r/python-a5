import numpy as np
import matplotlib.pyplot as plt

# ── 1. Load data from file ──────────────────────────────────────────
data = np.loadtxt("ex2data1.txt", delimiter=",")
X = data[:, :2]          # Features (columns 1 & 2)
y = data[:, 2]           # Target   (column 3)
m = len(y)                # Number of training examples

# ── 2. Feature normalization ────────────────────────────────────────
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# ── 3. Setup ────────────────────────────────────────────────────────
X_b = np.c_[np.ones(m), X_norm]   # Add intercept term (bias column of 1s)
theta = np.zeros(X_b.shape[1])    # Initialize theta to zeros
alpha = 0.01                       # Learning rate
iterations = 400                   # Number of gradient descent steps

# ── 4. Gradient descent ─────────────────────────────────────────────
def gradient_descent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        gradient = (1 / m) * (X.T @ (X @ theta - y))
        theta -= alpha * gradient
    return theta

# ── 5. Train and print result ───────────────────────────────────────
theta = gradient_descent(X_b, y, theta, alpha, iterations)
print("Learned theta:", theta)

# ── 6. Plot data + decision boundary ────────────────────────────────
pos = y == 1
neg = y == 0
plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='blue',  label='Admitted')
plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='red',   label='Not Admitted')

# Decision boundary: theta0 + theta1*x1_norm + theta2*x2_norm = 0.5
# Solve for x2 in original scale
x1_vals = np.array([X[:, 0].min(), X[:, 0].max()])
x1_norm = (x1_vals - X_mean[0]) / X_std[0]
x2_norm = (0.5 - theta[0] - theta[1] * x1_norm) / theta[2]
x2_vals = x2_norm * X_std[1] + X_mean[1]

plt.plot(x1_vals, x2_vals, 'g-', label='Decision boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Multivariate Linear Regression')
plt.legend()
plt.show()
