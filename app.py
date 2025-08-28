from flask import Flask, render_template, request
import numpy as np
import pickle
import os
from simple_model import SimpleLinearRegression

app = Flask(__name__)

# Feature names for the Boston house-prices dataset
FEATURE_NAMES = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]

# Vietnamese descriptions for each feature
FEATURE_LABELS = {
    'CRIM': 'Tỷ lệ tội phạm theo khu vực',
    'ZN': 'Tỷ lệ diện tích đất dành cho lô lớn (>25,000 ft²)',
    'INDUS': 'Tỷ lệ diện tích thương mại không bán lẻ (%)',
    'CHAS': 'Chạy dọc sông (1 nếu có, 0 nếu không)',
    'NOX': 'Nồng độ NOx (phần mười triệu)',
    'RM': 'Số phòng ngủ trung bình trong nhà',
    'AGE': 'Tỷ lệ nhà cũ (xây trước 1940) (%)',
    'DIS': 'Khoảng cách đến 5 trung tâm việc làm',
    'RAD': 'Chỉ số tiếp cận đường vành đai',
    'TAX': 'Thuế tài sản (mỗi $10,000)',
    'PTRATIO': 'Tỷ lệ học sinh trên giáo viên',
    'B': 'Chỉ báo dân số (1000(Bk - 0.63)^2)',
    'LSTAT': 'Tỷ lệ dân thu nhập thấp (%)'
}

# Default sample values
SAMPLE_VALUES = {
    'CRIM': 0.1, 'ZN': 0, 'INDUS': 8.0, 'CHAS': 0,
    'NOX': 0.5, 'RM': 6.0, 'AGE': 50.0, 'DIS': 4.0,
    'RAD': 4, 'TAX': 300, 'PTRATIO': 18.5, 'B': 390.0, 'LSTAT': 12.0
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')


def load_model():
    """Load the trained model from pickle file"""
    if not os.path.exists(MODEL_PATH):
        # If no saved model, create a default one
        print("No saved model found, creating default model")
        return SimpleLinearRegression()
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Return default model if loading fails
        return SimpleLinearRegression()


model = load_model()


@app.route('/')
def index():
    """Render home page with input form"""
    return render_template('index.html', 
                         features=FEATURE_NAMES, 
                         labels=FEATURE_LABELS, 
                         samples=SAMPLE_VALUES)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Extract feature values from form
        values = []
        for feat in FEATURE_NAMES:
            raw = request.form.get(feat, '').strip()
            values.append(float(raw) if raw else 0.0)

        # Make prediction
        arr = np.array(values).reshape(1, -1)
        
        # Check if model is trained
        if model.coef_ is None:
            return render_template('error.html', 
                                 message='Model not trained. Run save_model.py to train the model.')
        
        pred = model.predict(arr)[0]
        price = round(float(pred), 2)
        
        return render_template('result.html', price=price)
        
    except Exception as e:
        return render_template('error.html', message=str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
