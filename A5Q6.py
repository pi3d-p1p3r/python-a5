import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('digits_data.csv')
X = data.iloc[:, :-1].values  # Features (first 64 columns)
y = data.iloc[:, -1].values   # Labels (last column)

# Normalize features to [0, 1] range
X = X / 16.0

# Add bias term (column of ones)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function with L2 regularization
def compute_cost(X, y, theta, lambda_reg):
    m = len(y)
    h = sigmoid(X @ theta)
    reg_term = (lambda_reg / (2 * m)) * np.sum(theta[1:]**2)
    cost = (-1 / m) * (y @ np.log(h + 1e-10) + (1 - y) @ np.log(1 - h + 1e-10)) + reg_term
    return cost

# Gradient descent with regularization
def gradient_descent(X, y, theta, alpha, num_iters, lambda_reg):
    m = len(y)
    for _ in range(num_iters):
        h = sigmoid(X @ theta)
        gradient = (1 / m) * (X.T @ (h - y)) + (lambda_reg / m) * np.r_[0, theta[1:]]
        theta -= alpha * gradient
    return theta

# Train One-vs-All classifiers
def train_one_vs_all(X, y, num_labels, alpha, num_iters, lambda_reg):
    n = X.shape[1]  # Number of features including bias
    all_theta = np.zeros((num_labels, n))
    for c in range(num_labels):
        y_c = (y == c).astype(int)  # Binary labels for class c
        theta = np.zeros(n)
        theta = gradient_descent(X, y_c, theta, alpha, num_iters, lambda_reg)
        all_theta[c, :] = theta
    return all_theta

# Predict class labels
def predict_one_vs_all(all_theta, X):
    probs = sigmoid(X @ all_theta.T)
    return np.argmax(probs, axis=1)

# Training parameters
num_labels = 10  # Digits 0-9
alpha = 0.1      # Learning rate
num_iters = 400  # Number of iterations
lambda_reg = 0.1 # Regularization parameter

# Train the model
all_theta = train_one_vs_all(X, y, num_labels, alpha, num_iters, lambda_reg)

# Predict on training set
y_pred = predict_one_vs_all(all_theta, X)

# Compute accuracy
accuracy = np.mean(y_pred == y) * 100
print(f'Training Accuracy: {accuracy:.2f}%')

# Visualize sample digits
def plot_digits(X, y, y_pred, num_samples=10):
    indices = np.random.choice(len(y), num_samples, replace=False)
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        image = X[idx, 1:].reshape(8, 8)  # Exclude bias term
        plt.imshow(image, cmap='gray')
        plt.title(f'True: {y[idx]}\nPred: {y_pred[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('digit_samples.png')
    plt.close()

# Generate visualization
plot_digits(X, y, y_pred)
print("Sample digit images with true and predicted labels saved as 'digit_samples.png'")