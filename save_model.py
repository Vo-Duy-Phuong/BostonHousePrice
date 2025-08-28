import pandas as pd
import numpy as np
import pickle
import os
from simple_model import SimpleLinearRegression

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
