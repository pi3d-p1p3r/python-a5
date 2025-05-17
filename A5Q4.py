import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize # Using minimize for optimization instead of manual gradient descent for robustness

# --- 1. Generate Dataset ---
def generate_microchip_data(n_samples=118, noise_level=0.05, random_seed=42):
    """
    Generates a synthetic microchip dataset similar to those used in ML courses.
    Features (X) are two test results, typically ranging from -1 to 1.
    Labels (y) are 0 (fail) or 1 (pass).
    The decision boundary is non-linear.
    """
    np.random.seed(random_seed)
    X = np.random.rand(n_samples, 2) * 2 - 1  # Features between -1 and 1

    # Create a complex, somewhat circular decision boundary
    # y=1 if inside a combination of shapes, 0 otherwise
    y = np.zeros(n_samples)
    
    # A more typical microchip-like boundary (e.g., two circles)
    dist_sq_1 = X[:, 0]**2 + X[:, 1]**2
    dist_sq_2 = (X[:, 0] - 0.2)**2 + (X[:, 1] + 0.4)**2 # Another center
    
    # Pass if within a certain radius of origin, but not too close, OR within another specific region
    # This creates a more complex shape
    condition1 = (dist_sq_1 < 0.7**2) & (dist_sq_1 > 0.2**2) 
    condition2 = dist_sq_2 < 0.3**2
    
    y[(condition1) | (condition2)] = 1
    
    # Add some random noise to labels to make it non-perfectly separable
    if noise_level > 0:
        flip_indices = np.random.choice(n_samples, int(n_samples * noise_level), replace=False)
        y[flip_indices] = 1 - y[flip_indices]
        
    return X, y.astype(int)

# --- Sigmoid function ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- 2. Map Features ---
# We will use sklearn's PolynomialFeatures.
# For example, degree=2 for (x1, x2) gives (1, x1, x2, x1^2, x1*x2, x2^2)

# --- 3. Implement Regularized Cost Function and Gradient ---
def cost_function_reg(theta, X_poly, y, lambda_val):
    m = len(y)
    h = sigmoid(X_poly @ theta)
    
    # Prevent log(0) issues
    epsilon = 1e-7
    h = np.clip(h, epsilon, 1 - epsilon)
    
    cost = (-1/m) * (y.T @ np.log(h) + (1-y).T @ np.log(1-h))
    
    # Regularization term (do not regularize theta[0])
    reg_term = (lambda_val / (2*m)) * np.sum(theta[1:]**2)
    
    total_cost = cost + reg_term
    return total_cost

def gradient_reg(theta, X_poly, y, lambda_val):
    m = len(y)
    h = sigmoid(X_poly @ theta)
    
    gradient = (1/m) * (X_poly.T @ (h - y))
    
    # Regularization term for gradient (do not regularize theta[0])
    reg_grad_term = (lambda_val / m) * theta
    reg_grad_term[0] = 0 # Don't regularize the bias term
    
    total_gradient = gradient + reg_grad_term
    return total_gradient

# --- 4. Plot Decision Boundaries (and data) ---
def plot_decision_boundary(X, y, X_poly_mapper, theta, lambda_val, ax):
    """
    Plots the data points and the decision boundary.
    X: original features (before polynomial mapping)
    y: labels
    X_poly_mapper: the PolynomialFeatures object used for mapping
    theta: learned parameters
    lambda_val: regularization parameter
    ax: matplotlib axis object
    """
    # Plot data
    ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='o', label='Pass (y=1)', edgecolors='k', alpha=0.7)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='x', label='Fail (y=0)', s=50, alpha=0.7)

    # Create a grid of points to plot the decision boundary
    u_min, u_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    v_min, v_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    u_vals = np.linspace(u_min, u_max, 100)
    v_vals = np.linspace(v_min, v_max, 100)
    
    z = np.zeros((len(u_vals), len(v_vals)))

    for i in range(len(u_vals)):
        for j in range(len(v_vals)):
            # Map the grid point to polynomial features
            point_poly = X_poly_mapper.transform(np.array([[u_vals[i], v_vals[j]]]))
            z[j, i] = point_poly @ theta # z = theta.T @ X_poly_point

    # Plot the contour z=0 (decision boundary)
    ax.contour(u_vals, v_vals, z, levels=[0], linewidths=2, colors='green')
    
    ax.set_xlabel("Microchip Test 1 Score")
    ax.set_ylabel("Microchip Test 2 Score")
    ax.set_title(f"Decision Boundary (λ = {lambda_val})")
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)

# --- Main ---
if __name__ == "__main__":
    X_orig, y_orig = generate_microchip_data(n_samples=118, random_seed=10) # Seed for reproducibility

    # Plot initial data
    plt.figure(figsize=(7,6))
    plt.scatter(X_orig[y_orig==1, 0], X_orig[y_orig==1, 1], c='blue', marker='o', label='Pass (y=1)', edgecolors='k', alpha=0.7)
    plt.scatter(X_orig[y_orig==0, 0], X_orig[y_orig==0, 1], c='red', marker='x', label='Fail (y=0)', s=50, alpha=0.7)
    plt.xlabel("Microchip Test 1 Score")
    plt.ylabel("Microchip Test 2 Score")
    plt.title("Microchip Test Results Data")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # --- Task 1: Map features into polynomial terms (up to 6th degree) ---
    poly_degree = 6
    # include_bias=True adds a column of ones for the intercept term.
    # This means theta[0] will be the bias, and X_poly will have the intercept column.
    poly_features_mapper = PolynomialFeatures(degree=poly_degree, include_bias=True)
    X_poly = poly_features_mapper.fit_transform(X_orig)
    
    print(f"Original X shape: {X_orig.shape}")
    print(f"Polynomial mapped X shape (degree {poly_degree}): {X_poly.shape}") # Should be (n_samples, n_poly_features)

    # --- Task 2 & 3: Implement regularized cost/gradient and choose lambda values ---
    lambda_values = [0, 1, 100] # Regularization parameters to test
    
    # Prepare subplots for decision boundaries
    num_lambdas = len(lambda_values)
    fig, axes = plt.subplots(1, num_lambdas, figsize=(7 * num_lambdas, 6), sharey=True)
    if num_lambdas == 1: # Make axes iterable even if only one subplot
        axes = [axes]

    print("\nTraining models with different lambda values:")
    for i, lambda_val in enumerate(lambda_values):
        print(f"\n--- Training for λ = {lambda_val} ---")
        
        # Initialize theta
        initial_theta = np.zeros(X_poly.shape[1])
        
        # Optimize theta using scipy.optimize.minimize
        # This is generally more robust than manual gradient descent for complex problems.
        # We provide the cost function and the gradient (jac=jacobian).
        # 'TNC' or 'BFGS' are good methods for this type of problem.
        options = {'maxiter': 400} # Can increase if needed
        result = minimize(cost_function_reg, 
                          initial_theta, 
                          args=(X_poly, y_orig, lambda_val), 
                          method='TNC', # or 'BFGS', 'L-BFGS-B'
                          jac=gradient_reg, 
                          options=options)
        
        optimal_theta = result.x
        final_cost = result.fun
        
        print(f"Optimization successful: {result.success}")
        print(f"Final cost: {final_cost:.4f}")
        # print(f"Optimal theta (first 5 elements): {optimal_theta[:5]}")

        # --- Task 4: Plot the decision boundaries ---
        plot_decision_boundary(X_orig, y_orig, poly_features_mapper, optimal_theta, lambda_val, axes[i])

        # Calculate accuracy on the training set
        predictions_prob = sigmoid(X_poly @ optimal_theta)
        predictions = (predictions_prob >= 0.5).astype(int)
        accuracy = np.mean(predictions == y_orig) * 100
        axes[i].text(0.05, 0.05, f'Accuracy: {accuracy:.2f}%', transform=axes[i].transAxes, 
                     fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
        print(f"Training Accuracy: {accuracy:.2f}%")

    plt.tight_layout()
    plt.savefig("regularized_logistic_regression_boundaries.png")
    plt.show()

    