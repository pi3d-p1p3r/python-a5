import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ============================================================
#  STEP 1: Generate Non-Linear Dataset  y = x·sin(x) + noise
# ============================================================
def generate_data(n=30, seed=42):
    np.random.seed(seed)
    X = np.linspace(0.1, 2 * np.pi, n)
    y = X * np.sin(X) + np.random.randn(n) * 0.3
    return X.reshape(-1, 1), y


# ============================================================
#  STEP 2: Map Features to Polynomial Terms (degree p)
#    x -> [1, x, x², ..., xᵖ]
# ============================================================
def poly_features(X, degree):
    return np.column_stack([X**d for d in range(degree + 1)])


# ============================================================
#  STEP 3: Feature Normalization (skip bias column)
# ============================================================
def normalize(X_poly):
    mu = X_poly[:, 1:].mean(axis=0)
    sigma = X_poly[:, 1:].std(axis=0) + 1e-8
    X_norm = X_poly.copy()
    X_norm[:, 1:] = (X_poly[:, 1:] - mu) / sigma
    return X_norm, mu, sigma


def apply_normalize(X_poly, mu, sigma):
    X_norm = X_poly.copy()
    X_norm[:, 1:] = (X_poly[:, 1:] - mu) / sigma
    return X_norm


# ============================================================
#  STEP 4: Regularized Linear Regression Cost & Gradient
#    J = (1/2m) Σ(h-y)² + (λ/2m) Σθ_j²   (skip θ₀)
# ============================================================
def cost(theta, X, y, lam):
    m = len(y)
    h = X @ theta
    J = (1 / (2 * m)) * np.sum((h - y)**2)
    J += (lam / (2 * m)) * np.sum(theta[1:]**2)
    return J


def gradient(theta, X, y, lam):
    m = len(y)
    h = X @ theta
    grad = (1 / m) * (X.T @ (h - y))
    reg = (lam / m) * theta
    reg[0] = 0
    return grad + reg


# ============================================================
#  STEP 5: Train (Optimize θ)
# ============================================================
def train(X, y, lam):
    theta0 = np.zeros(X.shape[1])
    res = minimize(cost, theta0, args=(X, y, lam),
                   method='TNC', jac=gradient, options={'maxfun': 500})
    return res.x


# ============================================================
#  STEP 6: Learning Curves
#    Train on subsets of size 1..m, record train & CV error
# ============================================================
def learning_curves(X_train, y_train, X_cv, y_cv, lam):
    m = X_train.shape[0]
    train_err, cv_err = [], []

    for i in range(1, m + 1):
        theta = train(X_train[:i], y_train[:i], lam)
        train_err.append(cost(theta, X_train[:i], y_train[:i], 0))
        cv_err.append(cost(theta, X_cv, y_cv, 0))

    return train_err, cv_err


# ============================================================
#  STEP 7: Select Best λ Using Cross-Validation
# ============================================================
def select_lambda(X_train, y_train, X_cv, y_cv, lambdas):
    train_errors, cv_errors = [], []
    for lam in lambdas:
        theta = train(X_train, y_train, lam)
        train_errors.append(cost(theta, X_train, y_train, 0))
        cv_errors.append(cost(theta, X_cv, y_cv, 0))
    return train_errors, cv_errors


# ============================================================
#  STEP 8: Main — Run Everything
# ============================================================
if __name__ == "__main__":

    # ---- 8a. Generate & split data (60% train / 20% cv / 20% test) ----
    X_raw, y = generate_data(n=60)
    m = len(y)
    idx = np.random.RandomState(1).permutation(m)
    X_raw, y = X_raw[idx], y[idx]

    m_tr, m_cv = int(0.6 * m), int(0.2 * m)
    X_tr, y_tr = X_raw[:m_tr], y[:m_tr]
    X_cv, y_cv = X_raw[m_tr:m_tr + m_cv], y[m_tr:m_tr + m_cv]
    X_te, y_te = X_raw[m_tr + m_cv:], y[m_tr + m_cv:]

    # ---- 8b. Map to polynomial features (degree 8) ----
    degree = 8
    Xp_tr, mu, sigma = normalize(poly_features(X_tr, degree))
    Xp_cv = apply_normalize(poly_features(X_cv, degree), mu, sigma)
    Xp_te = apply_normalize(poly_features(X_te, degree), mu, sigma)
    print(f"Poly degree: {degree}  |  Feature count: {Xp_tr.shape[1]}")

    # ---- 8c. Train with λ=0 and show learned θ ----
    lam = 0
    theta = train(Xp_tr, y_tr, lam)
    print(f"\nLearned theta (lambda={lam}):\n{np.round(theta, 4)}")
    print(f"Train cost: {cost(theta, Xp_tr, y_tr, 0):.4f}")
    print(f"CV    cost: {cost(theta, Xp_cv, y_cv, 0):.4f}")

    # ---- 8d. PLOT 1 — Polynomial Fit on Data ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for ax, lam_val, title in zip(axes, [0, 1, 100],
                                  ["No Reg (λ=0) — Overfit",
                                   "Moderate (λ=1) — Good Fit",
                                   "Heavy Reg (λ=100) — Underfit"]):
        th = train(Xp_tr, y_tr, lam_val)
        x_plot = np.linspace(X_raw.min() - 0.3, X_raw.max() + 0.3, 200).reshape(-1, 1)
        xp_plot = apply_normalize(poly_features(x_plot, degree), mu, sigma)

        ax.scatter(X_tr, y_tr, c='blue', label='Train', edgecolors='k', zorder=3)
        ax.scatter(X_cv, y_cv, c='orange', marker='x', s=60, label='CV', zorder=3)
        ax.plot(x_plot, xp_plot @ th, 'r-', lw=2, label='Fit')
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize='small')
        ax.grid(True, ls='--', alpha=0.5)

    plt.suptitle("Polynomial Regression Fit (degree 8)", fontweight='bold')
    plt.tight_layout()
    plt.savefig("poly_fit.png")
    plt.show()

    # ---- 8e. PLOT 2 — Learning Curves (λ=1) ----
    lam_lc = 1
    t_err, c_err = learning_curves(Xp_tr, y_tr, Xp_cv, y_cv, lam_lc)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(t_err) + 1), t_err, 'b-', label='Train Error')
    plt.plot(range(1, len(c_err) + 1), c_err, 'r-', label='CV Error')
    plt.xlabel("Training Set Size")
    plt.ylabel("Error")
    plt.title(f"Learning Curves (lambda = {lam_lc})")
    plt.legend()
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("learning_curves.png")
    plt.show()

    # ---- 8f. PLOT 3 — Selecting λ (Bias-Variance Trade-off) ----
    lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    lam_tr, lam_cv = select_lambda(Xp_tr, y_tr, Xp_cv, y_cv, lambdas)

    best_idx = np.argmin(lam_cv)
    print(f"\nBest lambda: {lambdas[best_idx]}  (CV error = {lam_cv[best_idx]:.4f})")

    # Test set performance with best λ
    best_theta = train(Xp_tr, y_tr, lambdas[best_idx])
    print(f"Test cost with best lambda: {cost(best_theta, Xp_te, y_te, 0):.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, lam_tr, 'b-o', label='Train Error')
    plt.plot(lambdas, lam_cv, 'r-o', label='CV Error')
    plt.axvline(lambdas[best_idx], color='green', ls='--',
                label=f'Best λ = {lambdas[best_idx]}')
    plt.xlabel("Lambda (λ)")
    plt.ylabel("Error")
    plt.title("Selecting Lambda — Bias-Variance Trade-off")
    plt.xscale('symlog', linthresh=0.001)
    plt.legend()
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("lambda_selection.png")
    plt.show()

    # ---- 8g. Discussion ----
    print("\n" + "=" * 60)
    print("DISCUSSION — Bias-Variance Trade-off")
    print("=" * 60)
    print("""
lambda = 0   (No regularization)
  - Polynomial fits every noise point -> wiggly curve.
  - Train error ~ 0, but CV error is high.
  - HIGH VARIANCE -> OVERFITTING.

lambda = 1   (Moderate regularization)
  - Balances data fit vs. model complexity.
  - Train and CV errors converge reasonably.
  - GOOD GENERALIZATION.

lambda = 100 (Heavy regularization)
  - Theta values shrink toward zero -> nearly flat line.
  - Both train and CV errors are high.
  - HIGH BIAS -> UNDERFITTING.

Learning Curves:
  - Small training set: train error low, CV error high (gap = variance).
  - Large training set: both errors converge.
  - If they converge at a HIGH value -> high bias (need more features).
  - If gap persists -> high variance (need more data or regularization).
""")
