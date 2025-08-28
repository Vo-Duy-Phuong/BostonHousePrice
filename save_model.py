# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import pickle
# import os

# CSV = 'boston_house_prices.csv'
# MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')


# def main():
#     df = pd.read_csv(CSV)
#     # Expect last column MEDV as price
#     if 'MEDV' not in df.columns:
#         raise RuntimeError('CSV must contain MEDV column')

#     X = df.drop('MEDV', axis=1)
#     y = df['MEDV']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)

#     lm = LinearRegression()
#     lm.fit(X_train, y_train)

#     y_pred = lm.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print(f'Model trained. MSE={mse:.4f}  R2={r2:.4f}')

#     with open(MODEL_PATH, 'wb') as f:
#         pickle.dump(lm, f)

#     print(f'Model saved to {MODEL_PATH}')


# if __name__ == '__main__':
#     main()
