# Boston House Price - Flask Web App

This project converts the Boston House Prices model into a small Flask web app.

Steps to run (Windows cmd):

1. Create a virtual environment (optional but recommended):

```
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Train and save the model:

```
python save_model.py
```

4. Run the Flask app:

```
python app.py
```

5. Open http://localhost:5000 in your browser.

Notes:
- The app expects `boston_house_prices.csv` to contain a `MEDV` column (target).
- The form fields correspond to the dataset features. Empty fields are treated as 0.
