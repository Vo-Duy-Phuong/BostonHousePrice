import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        # Add bias column
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        # Normal equation: θ = (X^T X)^(-1) X^T y
        theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_
