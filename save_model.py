import pandas as pd
import numpy as np
import pickle
import os

# Simple Linear Regression class using numpy
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

CSV = 'boston_house_prices.csv'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')


def main():
    df = pd.read_csv(CSV)
    # Expect last column MEDV as price
    if 'MEDV' not in df.columns:
        raise RuntimeError('CSV must contain MEDV column')

    X = df.drop('MEDV', axis=1).values
    y = df['MEDV'].values

    # Train simple model
    lm = SimpleLinearRegression()
    lm.fit(X, y)

    # Test prediction
    pred = lm.predict(X[:5])
    print(f'Model trained. Sample predictions: {pred[:3]}')

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(lm, f)

    print(f'Model saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()
