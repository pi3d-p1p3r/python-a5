import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate random dataset
def generate_data(n_samples=100):
    # Parameters to create a separation between admitted and not admitted
    # Students with higher scores are more likely to be admitted
    mean_admitted = [85, 80]
    cov_admitted = [[40, 10], [10, 35]]
    
    mean_not_admitted = [65, 60]
    cov_not_admitted = [[30, 8], [8, 25]]
    
    # Generate half positive examples (admitted)
    n_positive = n_samples // 2
    X_positive = np.random.multivariate_normal(mean_admitted, cov_admitted, n_positive)
    y_positive = np.ones(n_positive)
    
    # Generate half negative examples (not admitted)
    n_negative = n_samples - n_positive
    X_negative = np.random.multivariate_normal(mean_not_admitted, cov_not_admitted, n_negative)
    y_negative = np.zeros(n_negative)
    
    # Combine the data
    X = np.vstack((X_positive, X_negative))
    y = np.hstack((y_positive, y_negative))
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y

# Implement sigmoid function
def sigmoid(z):
    """
    Compute the sigmoid function for the input z
    """
    return 1.0 / (1.0 + np.exp(-z))

# Implement cost function for logistic regression
def compute_cost(X, y, theta):
    """
    Compute the cost of using theta as the parameter for logistic regression
    """
    m = len(y)
    h = sigmoid(X @ theta)
    
    # Handle potential numerical issues to avoid log(0)
    epsilon = 1e-5
    h = np.clip(h, epsilon, 1-epsilon)
    
    cost = (-1/m) * (y @ np.log(h) + (1-y) @ np.log(1-h))
    return cost

# Implement gradient descent
def gradient_descent(X, y, theta, alpha, num_iterations):
    """
    Optimize theta using gradient descent
    """
    m = len(y)
    J_history = []
    
    for i in range(num_iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta = theta - alpha * gradient
        
        # Save the cost for every iteration
        J_history.append(compute_cost(X, y, theta))
        
    return theta, J_history

# Predict using the optimized theta
def predict(X, theta):
    """
    Predict whether the label is 0 or 1 using the learned logistic regression model
    """
    return sigmoid(X @ theta) >= 0.5

# Plot decision boundary as a line - FIXED FUNCTION (assumed correct by user, and it is)
def plot_decision_boundary(X, y, theta):
    """
    Plot the data points and the decision boundary as a line
    """
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    admitted = y == 1
    not_admitted = y == 0
    
    plt.scatter(X[admitted, 0], X[admitted, 1], c='green', marker='o', 
                label='Admitted', edgecolors='k')
    plt.scatter(X[not_admitted, 0], X[not_admitted, 1], c='red', marker='x', 
                label='Not Admitted', edgecolors='k') # Original had edgecolors='k', I'll keep it, though image shows no edge for 'x'
    
    # Draw the decision boundary line
    # The decision boundary is where theta[0] + theta[1]*x1 + theta[2]*x2 = 0
    # Solving for x2: x2 = (-theta[0] - theta[1]*x1) / theta[2]
    
    if abs(theta[2]) > 1e-10:  # Avoid division by near-zero values
        x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
        x_vals = np.array([x_min, x_max])
        y_vals = (-theta[0] - theta[1] * x_vals) / theta[2]
        
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='Decision Boundary')
    
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.title('Student Admission Prediction')
    plt.legend()
    plt.grid(True)
    plt.axis([45, 100, 45, 105])  # Set axis limits to match the example image
    plt.savefig('decision_boundary.png')
    plt.show()

# Main function
def main():
    # Generate the dataset
    X, y = generate_data(100)
    
    # Visualize data using a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(X[y==1, 0], X[y==1, 1], c='green', marker='o', label='Admitted', edgecolors='k')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='x', label='Not Admitted', edgecolors='k') # Corrected edgecolors here for consistency if desired
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.title('Student Admission Data')
    plt.legend()
    plt.grid(True)
    plt.axis([45, 100, 45, 105])  # Set axis limits to match the example image
    plt.savefig('data_visualization.png')
    plt.show()
    
    # --- FIX: Implement Feature Scaling ---
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    X_scaled = (X - mean_X) / std_X
    
    # Add intercept term to X_scaled
    X_train_scaled_intercept = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))
    
    # Initialize theta parameters
    initial_theta = np.zeros(X_train_scaled_intercept.shape[1])
    
    # Hyperparameters for gradient descent
    # With feature scaling, we can use a larger learning rate and/or fewer iterations.
    # Let's try alpha = 1.0, which works well for scaled features.
    alpha = 1.0 # Adjusted alpha
    num_iterations = 1000 # Kept original number of iterations
    
    # Run gradient descent on scaled features
    optimal_theta_scaled, J_history = gradient_descent(X_train_scaled_intercept, y, initial_theta, alpha, num_iterations)
    
    print(f"Optimal theta (scaled features): {optimal_theta_scaled}")

    # --- FIX: Transform theta back to original scale for plotting ---
    # z = th0_sc + th1_sc * ((x1 - m1)/s1) + th2_sc * ((x2 - m2)/s2)
    # z = (th0_sc - th1_sc*m1/s1 - th2_sc*m2/s2) + (th1_sc/s1)*x1 + (th2_sc/s2)*x2
    # So, th0_orig = th0_sc - th1_sc*m1/s1 - th2_sc*m2/s2
    #     th1_orig = th1_sc/s1
    #     th2_orig = th2_sc/s2
    theta_0_orig = optimal_theta_scaled[0] - (optimal_theta_scaled[1] * mean_X[0] / std_X[0]) - (optimal_theta_scaled[2] * mean_X[1] / std_X[1])
    theta_1_orig = optimal_theta_scaled[1] / std_X[0]
    theta_2_orig = optimal_theta_scaled[2] / std_X[1]
    optimal_theta_orig = np.array([theta_0_orig, theta_1_orig, theta_2_orig])
    
    print(f"Optimal theta (original features for plotting): {optimal_theta_orig}")
    
    # Plot the convergence of cost function
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_iterations), J_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost J')
    plt.title('Convergence of Cost Function')
    plt.grid(True)
    plt.savefig('cost_convergence.png')
    plt.show()
    
    # Plot the decision boundary using original X and transformed theta
    plot_decision_boundary(X, y, optimal_theta_orig)
    
    # Evaluate accuracy on the training set (using scaled features for prediction)
    predictions = predict(X_train_scaled_intercept, optimal_theta_scaled)
    accuracy = accuracy_score(y, predictions)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
    
    # Show example predictions
    print("\nExample predictions:")
    print("Exam 1 Score | Exam 2 Score | Predicted | Actual")
    for i in range(5): # Using original X for displaying scores
        # Prediction for example must use scaled features if model was trained on them
        example_x_scaled = (X[i,:] - mean_X) / std_X
        example_x_with_intercept = np.hstack((np.ones(1), example_x_scaled))
        example_prediction = predict(example_x_with_intercept.reshape(1,-1), optimal_theta_scaled)[0]
        print(f"{X[i, 0]:.2f} | {X[i, 1]:.2f} | {int(example_prediction)} | {int(y[i])}")
        
if __name__ == "__main__":
    main()