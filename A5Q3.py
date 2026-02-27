import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  STEP 1: Generate Student Admission Dataset
#    - 2 exam scores (features), binary admit/reject labels
# ============================================================
def generate_data(n=100):
    np.random.seed(42)
    n_pos = n // 2

    X_pos = np.random.multivariate_normal([85, 80], [[40, 10], [10, 35]], n_pos)
    X_neg = np.random.multivariate_normal([65, 60], [[30,  8], [ 8, 25]], n - n_pos)

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n_pos), np.zeros(n - n_pos)])

    idx = np.random.permutation(n)
    return X[idx], y[idx]

# ============================================================
#  STEP 2: Feature Scaling (zero-mean, unit-variance)
# ============================================================
def scale_features(X):
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    return (X - mu) / sigma, mu, sigma

# ============================================================
#  STEP 3: Sigmoid Function
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ============================================================
#  STEP 4: Cost Function
#    J = -(1/m) [y·log(h) + (1-y)·log(1-h)]
# ============================================================
def cost(theta, X, y):
    m = len(y)
    h = np.clip(sigmoid(X @ theta), 1e-7, 1 - 1e-7)
    return (-1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))

# ============================================================
#  STEP 5: Gradient Descent
# ============================================================
def gradient_descent(X, y, alpha=1.0, iters=1000):
    theta = np.zeros(X.shape[1])
    history = []

    for _ in range(iters):
        grad = (1 / len(y)) * (X.T @ (sigmoid(X @ theta) - y))
        theta -= alpha * grad
        history.append(cost(theta, X, y))

    return theta, history

# ============================================================
#  STEP 6: Convert Scaled θ Back to Original Feature Scale
#    z = θ₀' + θ₁'·(x₁-μ₁)/σ₁ + θ₂'·(x₂-μ₂)/σ₂
#    → θ₀ = θ₀' - θ₁'·μ₁/σ₁ - θ₂'·μ₂/σ₂
#    → θⱼ = θⱼ'/σⱼ
# ============================================================
def unscale_theta(theta_sc, mu, sigma):
    theta0 = theta_sc[0] - np.sum(theta_sc[1:] * mu / sigma)
    return np.hstack([theta0, theta_sc[1:] / sigma])

# ============================================================
#  STEP 7: Plot Decision Boundary
# ============================================================
def plot_boundary(X, y, theta, ax):
    ax.scatter(X[y == 1, 0], X[y == 1, 1],
               c='green', marker='o', label='Admitted', edgecolors='k')
    ax.scatter(X[y == 0, 0], X[y == 0, 1],
               c='red', marker='x', label='Not Admitted', s=50)

    # Boundary: θ₀ + θ₁x₁ + θ₂x₂ = 0  →  x₂ = -(θ₀ + θ₁x₁) / θ₂
    if abs(theta[2]) > 1e-10:
        x1 = np.array([X[:, 0].min() - 5, X[:, 0].max() + 5])
        x2 = -(theta[0] + theta[1] * x1) / theta[2]
        ax.plot(x1, x2, 'b-', lw=2, label='Decision Boundary')

    ax.set_xlabel("Exam 1 Score")
    ax.set_ylabel("Exam 2 Score")
    ax.legend(fontsize='small')
    ax.grid(True, ls='--', alpha=0.6)
    ax.axis([45, 100, 45, 105])

# ============================================================
#  STEP 8: Main — Run Everything
# ============================================================
if __name__ == "__main__":

    # ---- 8a. Generate data ----
    X, y = generate_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # ---- 8b. Scale features & add bias ----
    X_sc, mu, sigma = scale_features(X)
    X_train = np.column_stack([np.ones(len(y)), X_sc])   # add bias column

    # ---- 8c. Train via gradient descent ----
    alpha, iters = 1.0, 1000
    theta_sc, J_hist = gradient_descent(X_train, y, alpha, iters)

    theta_orig = unscale_theta(theta_sc, mu, sigma)
    print(f"\nTheta (scaled) : {np.round(theta_sc,  4)}")
    print(f"Theta (original): {np.round(theta_orig, 4)}")

    # ---- 8d. Training accuracy ----
    preds = (sigmoid(X_train @ theta_sc) >= 0.5).astype(int)
    acc = np.mean(preds == y) * 100
    print(f"\nTraining Accuracy: {acc:.2f}%")

    # ---- 8e. Example predictions ----
    print("\nExam 1  | Exam 2  | Predicted | Actual")
    print("-" * 45)
    for i in range(5):
        x_sc = np.hstack([1, (X[i] - mu) / sigma])
        pred = int(sigmoid(x_sc @ theta_sc) >= 0.5)
        print(f" {X[i,0]:6.2f} | {X[i,1]:6.2f} |     {pred}     |   {int(y[i])}")

    # ---- 8f. Plots ----
    # --------------------------------------------------------
    # Figure 1: Cost Convergence
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(J_hist, 'b-')
    plt.xlabel("Iterations")
    plt.ylabel("Cost J")
    plt.title("Cost Convergence")
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("cost_convergence.png")
    plt.show()

    # --------------------------------------------------------
    # Figure 2: Data & Decision Boundary (1x2 Subplot)
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Plot 1 — Scatter
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1],
                    c='green', marker='o', label='Admitted', edgecolors='k')
    axes[0].scatter(X[y == 0, 0], X[y == 0, 1],
                    c='red', marker='x', label='Not Admitted', s=50)
    axes[0].set_xlabel("Exam 1 Score")
    axes[0].set_ylabel("Exam 2 Score")
    axes[0].set_title("Student Admission Data")
    axes[0].legend(fontsize='small')
    axes[0].grid(True, ls='--', alpha=0.6)
    axes[0].axis([45, 100, 45, 105])

    # Plot 2 — Decision boundary
    plot_boundary(X, y, theta_orig, axes[1])
    axes[1].set_title("Decision Boundary")
    plt.suptitle("Logistic Regression — Student Admission",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("admission_results.png")
    plt.show()
