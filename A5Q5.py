import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
def generate_data(n_samples=100):
    x = np.linspace(0, 10, n_samples)
    y = x * np.sin(x) + np.random.normal(0, 1, n_samples)  # y = x * sin(x) + noise
    return x.reshape(-1, 1), y

# Transform features to polynomial terms
def transform_features(x, degree):
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(x)

# Train Ridge regression model and compute errors
def train_and_evaluate(x_train, y_train, x_cv, y_cv, degree, lambda_val):
    x_train_poly = transform_features(x_train, degree)
    x_cv_poly = transform_features(x_cv, degree)
    
    model = Ridge(alpha=lambda_val)
    model.fit(x_train_poly, y_train)
    
    y_train_pred = model.predict(x_train_poly)
    y_cv_pred = model.predict(x_cv_poly)
    
    train_error = mean_squared_error(y_train, y_train_pred)
    cv_error = mean_squared_error(y_cv, y_cv_pred)
    
    return train_error, cv_error, model

# Plot learning curves for different lambda values
def plot_learning_curves(x, y, degree, lambda_vals):
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.2, random_state=42)
    
    training_sizes = np.linspace(5, len(x_train), 10, dtype=int)
    
    plt.figure(figsize=(10, 6))
    for lambda_val in lambda_vals:
        train_errors = []
        cv_errors = []
        
        for size in training_sizes:
            x_train_subset = x_train[:size]
            y_train_subset = y_train[:size]
            
            train_error, cv_error, _ = train_and_evaluate(x_train_subset, y_train_subset, x_cv, y_cv, degree, lambda_val)
            train_errors.append(train_error)
            cv_errors.append(cv_error)
        
        plt.plot(training_sizes, train_errors, label=f'Train (λ={lambda_val})')
        plt.plot(training_sizes, cv_errors, label=f'CV (λ={lambda_val})')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curves for Different λ')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot polynomial regression fit
def plot_regression_fit(x, y, degree, lambda_val):
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.2, random_state=42)
    
    _, _, model = train_and_evaluate(x_train, y_train, x_cv, y_cv, degree, lambda_val)
    
    x_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    x_plot_poly = transform_features(x_plot, degree)
    y_plot = model.predict(x_plot_poly)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data', alpha=0.5)
    plt.plot(x_plot, y_plot, color='red', label='Polynomial Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Polynomial Regression Fit (Degree={degree}, λ={lambda_val})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate data
    x, y = generate_data(n_samples=100)
    
    # Set parameters
    degree = 5  # Polynomial degree
    lambda_vals = [0, 0.1, 1, 10]  # Regularization parameters to try
    
    # Plot learning curves
    plot_learning_curves(x, y, degree, lambda_vals)
    
    # Plot regression fit for lambda=1
    plot_regression_fit(x, y, degree, lambda_val=1)

    # Display learned parameters for lambda=1
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.2, random_state=42)
    _, _, model = train_and_evaluate(x_train, y_train, x_cv, y_cv, degree, lambda_val=1)
    print("Learned parameters (θ) for λ=1:", model.coef_)