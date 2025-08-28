import pandas as pd
import numpy as np
import pickle
import os
from simple_model import SimpleLinearRegression

CSV = 'boston_house_prices.csv'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')


def main():
    """Train and save the linear regression model"""
    # Load data
    df = pd.read_csv(CSV)
    if 'MEDV' not in df.columns:
        raise RuntimeError('CSV must contain MEDV column')

    X = df.drop('MEDV', axis=1).values
    y = df['MEDV'].values

    # Train model
    model = SimpleLinearRegression()
    model.fit(X, y)

    # Test prediction
    pred = model.predict(X[:5])
    print(f'Model trained successfully!')
    print(f'Sample predictions: {pred[:3]}')

    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f'Model saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()
