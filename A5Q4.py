import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize

# ============================================================
#  STEP 1: Generate Microchip Dataset
#  - 2 test scores (features) with pass/fail labels
#  - Non-linearly separable data
# ============================================================
def generate_data(n=118, seed=10):
    np.random.seed(seed)
    X = np.random.rand(n, 2) * 2 - 1  # scores between -1 and 1

    # Create a non-linear boundary (ring + blob shape)
    r1 = X[:, 0]**2 + X[:, 1]**2
    r2 = (X[:, 0] - 0.2)**2 + (X[:, 1] + 0.4)**2

    y = np.zeros(n, dtype=int)
    y[((r1 < 0.49) & (r1 > 0.04)) | (r2 < 0.09)] = 1  # pass

    # Flip ~5% of labels so it's not perfectly separable
    flip = np.random.choice(n, int(n * 0.05), replace=False)
    y[flip] = 1 - y[flip]
    return X, y


# ============================================================
#  STEP 2: Map Features to Polynomial Terms (up to degree 6)
# ============================================================
def map_features(X, degree=6):
    mapper = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = mapper.fit_transform(X)
    return X_poly, mapper


# ============================================================
#  STEP 3: Sigmoid Function
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ============================================================
#  STEP 4: Regularized Cost Function
#    J = -(1/m) * [y·log(h) + (1-y)·log(1-h)]
#        + (λ / 2m) * Σθ_j²   (j = 1,2,...  skip θ₀)
# ============================================================
def cost(theta, X_poly, y, lam):
    m = len(y)
    h = sigmoid(X_poly @ theta)
    h = np.clip(h, 1e-7, 1 - 1e-7)           # avoid log(0)

    J = (-1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    J += (lam / (2 * m)) * np.sum(theta[1:]**2)  # regularization (skip θ₀)
    return J


# ============================================================
#  STEP 5: Regularized Gradient
#    ∂J/∂θ_j = (1/m) * Σ(h-y)·x_j  +  (λ/m)·θ_j   (j ≥ 1)
#    ∂J/∂θ_0 = (1/m) * Σ(h-y)·x_0                   (no reg)
# ============================================================
def gradient(theta, X_poly, y, lam):
    m = len(y)
    h = sigmoid(X_poly @ theta)

    grad = (1 / m) * (X_poly.T @ (h - y))
    reg = (lam / m) * theta
    reg[0] = 0                                 # don't regularize θ₀
    return grad + reg


# ============================================================
#  STEP 6: Train (Optimize θ) Using scipy.optimize.minimize
# ============================================================
def train(X_poly, y, lam):
    theta0 = np.zeros(X_poly.shape[1])
    res = minimize(cost, theta0,
                   args=(X_poly, y, lam),
                   method='TNC',
                   jac=gradient,
                   options={'maxfun': 400})
    return res.x, res.fun


# ============================================================
#  STEP 7: Plot Decision Boundary
# ============================================================
def plot_boundary(X, y, mapper, theta, lam, ax):
    # Scatter the data
    ax.scatter(X[y == 1, 0], X[y == 1, 1],
               c='blue', marker='o', label='Pass', edgecolors='k', alpha=0.7)
    ax.scatter(X[y == 0, 0], X[y == 0, 1],
               c='red', marker='x', label='Fail', s=50, alpha=0.7)

    # Build a grid and evaluate θᵀx over it
    u = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 100)
    v = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 100)
    z = np.zeros((len(v), len(u)))

    for i in range(len(u)):
        for j in range(len(v)):
            point = mapper.transform([[u[i], v[j]]])
            z[j, i] = (point @ theta).item()

    # Shade prediction regions so they are always visible
    ax.contourf(u, v, z, levels=[-1e10, 0, 1e10],
                colors=['#ffcccc', '#ccccff'], alpha=0.3)

    # Draw boundary line at z=0 (only if z spans both sides)
    if z.min() < 0 < z.max():
        ax.contour(u, v, z, levels=[0], linewidths=2, colors='green')

    ax.set_xlabel("Test 1 Score")
    ax.set_ylabel("Test 2 Score")
    ax.set_title(f"lambda = {lam}")
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)


# ============================================================
#  STEP 8: Main — Run Everything
# ============================================================
if __name__ == "__main__":

    # ---- 8a. Generate data ----
    X, y = generate_data()

    # ---- 8b. Map to polynomial features (degree 6) ----
    X_poly, mapper = map_features(X, degree=6)
    print(f"Original shape : {X.shape}")
    print(f"Poly-mapped shape: {X_poly.shape}")

    # ---- 8c. Choose λ values and train ----
    lambdas = [0, 1, 100]   # lambda values to compare
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)

    for i, lam in enumerate(lambdas):
        theta, final_cost = train(X_poly, y, lam)

        # Training accuracy
        preds = (sigmoid(X_poly @ theta) >= 0.5).astype(int)
        acc = np.mean(preds == y) * 100

        print(f"\nlambda = {lam:>3}  |  Cost = {final_cost:.4f}  |  Accuracy = {acc:.2f}%")

        # ---- 8d. Plot decision boundary ----
        plot_boundary(X, y, mapper, theta, lam, axes[i])
        axes[i].text(0.05, 0.05, f'Acc: {acc:.1f}%',
                     transform=axes[i].transAxes, fontsize=10,
                     bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))

    plt.suptitle("Regularization in Logistic Regression — Decision Boundaries",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("regularization_results.png")
    plt.show()

    # ---- 8e. Discussion ----
    print("\n" + "=" * 60)
    print("DISCUSSION - Effect of lambda on Underfitting / Overfitting")
    print("=" * 60)
    print("""
lambda = 0   (No regularization)
  - The model tries to fit every data point.
  - Decision boundary is very complex (wiggly).
  - HIGH VARIANCE -> OVERFITTING.

lambda = 1   (Moderate regularization)
  - Balances fitting the data vs. keeping theta small.
  - Smooth, reasonable decision boundary.
  - GOOD GENERALIZATION (best trade-off).

lambda = 100 (Heavy regularization)
  - Theta values are pushed very close to zero.
  - Decision boundary is nearly a straight line (too simple).
  - HIGH BIAS -> UNDERFITTING.
""")
