import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression


# ============================================================
#  STEP 1: Load & Prepare the Digits Dataset (0-9)
# ============================================================
def load_data():
    digits = load_digits()                      # 8x8 images, 10 classes
    X, y = digits.data, digits.target           # X: (1797,64)  y: (1797,)
    X = X / 16.0                                # normalize pixel values to [0,1]
    X = np.column_stack([np.ones(len(y)), X])   # add bias column -> (1797,65)
    return X, y, digits.images


# ============================================================
#  STEP 2: Sigmoid Function
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ============================================================
#  STEP 3: Regularized Cost & Gradient
#    J = -(1/m)[y·log(h) + (1-y)·log(1-h)] + (λ/2m)Σθ_j²
# ============================================================
def cost(theta, X, y, lam):
    m = len(y)
    h = sigmoid(X @ theta)
    h = np.clip(h, 1e-7, 1 - 1e-7)
    J = (-1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    J += (lam / (2 * m)) * np.sum(theta[1:]**2)
    return J


def gradient(theta, X, y, lam):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (1 / m) * (X.T @ (h - y))
    reg = (lam / m) * theta
    reg[0] = 0                                  # don't regularize θ₀
    return grad + reg


# ============================================================
#  STEP 4: One-vs-All Training
#    For each class k: y_k = 1 if y==k else 0, train θ_k
# ============================================================
def train_ova(X, y, num_classes, lam=0.1):
    n = X.shape[1]
    all_theta = np.zeros((num_classes, n))       # (K x n)

    for k in range(num_classes):
        y_k = (y == k).astype(float)             # binary labels for class k
        theta0 = np.zeros(n)
        res = minimize(cost, theta0, args=(X, y_k, lam),
                       method='TNC', jac=gradient,
                       options={'maxfun': 400})
        all_theta[k] = res.x
        print(f"  Class {k}: cost = {res.fun:.4f}")

    return all_theta


# ============================================================
#  STEP 5: Predict — Pick Class With Highest Probability
# ============================================================
def predict(X, all_theta):
    probs = sigmoid(X @ all_theta.T)             # (m x K)
    return np.argmax(probs, axis=1)


# ============================================================
#  STEP 6: Visualize Sample Digits
# ============================================================
def plot_samples(images, y, n_rows=2, n_cols=8):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5))
    indices = np.random.choice(len(y), n_rows * n_cols, replace=False)

    for ax, idx in zip(axes.ravel(), indices):
        ax.imshow(images[idx], cmap='gray_r', interpolation='nearest')
        ax.set_title(f"{y[idx]}", fontsize=10)
        ax.axis('off')

    plt.suptitle("Sample Digits from Dataset", fontweight='bold')
    plt.tight_layout()
    plt.savefig("digit_samples.png")
    plt.show()


# ============================================================
#  STEP 7: Visualize Class Distribution
# ============================================================
def plot_distribution(y, num_classes):
    counts = [np.sum(y == k) for k in range(num_classes)]
    plt.figure(figsize=(8, 4))
    plt.bar(range(num_classes), counts, color='steelblue', edgecolor='k')
    plt.xlabel("Digit Class")
    plt.ylabel("Count")
    plt.title("Class Distribution in Digits Dataset")
    plt.xticks(range(num_classes))
    plt.grid(axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.show()


# ============================================================
#  STEP 8: Main — Run Everything
# ============================================================
if __name__ == "__main__":

    num_classes = 10

    # ---- 8a. Load data & show samples ----
    X, y, images = load_data()
    print(f"Dataset shape: X={X.shape}  y={y.shape}  Classes={num_classes}")

    plot_samples(images, y)
    plot_distribution(y, num_classes)

    # ---- 8b. Train One-vs-All classifiers ----
    lam = 0.1
    print(f"\nTraining {num_classes} OvA classifiers (lambda={lam})...")
    all_theta = train_ova(X, y, num_classes, lam)

    # ---- 8c. Predict & compute accuracy ----
    preds = predict(X, all_theta)
    acc = np.mean(preds == y) * 100
    print(f"\nTraining Accuracy (OvA): {acc:.2f}%")

    # ---- 8d. Show learned θ stats per class ----
    print(f"\nLearned Parameters — theta shape: {all_theta.shape}")
    for k in range(num_classes):
        th = all_theta[k]
        print(f"  Class {k}: min={th.min():.4f}  max={th.max():.4f}  "
              f"norm={np.linalg.norm(th):.4f}")

    # ---- 8e. Scikit-learn comparison ----
    print("\n--- Scikit-Learn Comparison ---")
    clf = LogisticRegression(max_iter=1000, C=1/lam)
    clf.fit(X[:, 1:], y)                         # sklearn adds bias internally
    sk_acc = clf.score(X[:, 1:], y) * 100
    print(f"Sklearn OvR Accuracy: {sk_acc:.2f}%")

    # ---- 8f. Visualize some predictions ----
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
    indices = np.random.RandomState(7).choice(len(y), 16, replace=False)

    for ax, idx in zip(axes.ravel(), indices):
        ax.imshow(images[idx], cmap='gray_r', interpolation='nearest')
        color = 'green' if preds[idx] == y[idx] else 'red'
        ax.set_title(f"P:{preds[idx]} T:{y[idx]}", fontsize=9, color=color)
        ax.axis('off')

    plt.suptitle("Predictions (P) vs True Labels (T)", fontweight='bold')
    plt.tight_layout()
    plt.savefig("ova_predictions.png")
    plt.show()

    # ---- 8g. Discussion ----
    print("\n" + "=" * 60)
    print("DISCUSSION — One-vs-All Logistic Regression")
    print("=" * 60)
    print("""
One-vs-All (OvA) Strategy:
  - Train K binary classifiers, one per class.
  - Classifier k learns: P(y=k | x) vs P(y≠k | x).
  - At prediction, pick class with highest probability.

Key Observations:
  - With 10 classes and 64 features, OvA achieves ~95%+ accuracy.
  - Regularization (lambda) prevents overfitting on training data.
  - Scikit-learn's LogisticRegression(multi_class='ovr') does
    the same thing but with more optimized solvers.

Advantages:
  - Simple to implement — just K binary problems.
  - Each classifier is independent (can train in parallel).

Limitations:
  - Assumes classes are separable in a one-vs-rest manner.
  - Softmax (multinomial) regression may work better when
    classes are mutually exclusive and tightly clustered.
""")
