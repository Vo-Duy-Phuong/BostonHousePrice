from flask import Flask, render_template, request
import numpy as np
import pickle
import os
from simple_model import SimpleLinearRegression

# Flask app to serve house price predictions
app = Flask(__name__)

# Path to model file  
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Feature names for the Boston house-prices dataset (used by the form)
FEATURE_NAMES = [
	'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
	'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]

# Vietnamese short descriptions for each feature (shown on the form)
FEATURE_LABELS = {
	'CRIM': 'Tỷ lệ tội phạm theo khu vực',
	'ZN': 'Tỷ lệ diện tích đất dành cho lô lớn (>25,000 ft²)',
	'INDUS': 'Tỷ lệ diện tích thương mại không bán lẻ (%)',
	'CHAS': 'Chạy dọc sông (1 nếu có, 0 nếu không)',
	'NOX': 'Nồng độ NOx (phần mười triệu)',
	'RM': 'Số phòng ngủ trung bình trong nhà',
	'AGE': 'Tỷ lệ nhà cũ (xây trước 1940) (%)',
	'DIS': 'Khoảng cách đến 5 trung tâm việc làm (giá trị khoảng cách)',
	'RAD': 'Chỉ số tiếp cận đường vành đai',
	'TAX': 'Thuế tài sản (mỗi $10,000)',
	'PTRATIO': 'Tỷ lệ học sinh trên giáo viên',
	'B': 'Chỉ báo dân số (1000(Bk - 0.63)^2)',
	'LSTAT': 'Tỷ lệ dân thu nhập thấp (%)'
}

# Convenient sample (default) values to quickly populate the form in the UI
SAMPLE_VALUES = {
	'CRIM': 0.1,
	'ZN': 0,
	'INDUS': 8.0,
	'CHAS': 0,
	'NOX': 0.5,
	'RM': 6.0,
	'AGE': 50.0,
	'DIS': 4.0,
	'RAD': 4,
	'TAX': 300,
	'PTRATIO': 18.5,
	'B': 390.0,
	'LSTAT': 12.0
}

# Path to model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')


def load_model():
	if not os.path.exists(MODEL_PATH):
		return None
	with open(MODEL_PATH, 'rb') as f:
		return pickle.load(f)


model = load_model()


@app.route('/', methods=['GET'])
def index():
	# Render a clean form for user to enter feature values
	return render_template('index.html', features=FEATURE_NAMES, labels=FEATURE_LABELS, samples=SAMPLE_VALUES)


@app.route('/predict', methods=['POST'])
def predict():
	global model
	if model is None:
		return render_template('error.html', message='Model not found. Run save_model.py to create model.pkl')

	# Read form inputs; fall back to 0 or reasonable defaults
	try:
		values = []
		for feat in FEATURE_NAMES:
			raw = request.form.get(feat)
			if raw is None or raw.strip() == '':
				# empty -> use median-like default 0
				val = 0.0
			else:
				val = float(raw)
			values.append(val)

		arr = np.array(values).reshape(1, -1)
		pred = model.predict(arr)[0]
		price = round(float(pred), 2)
		return render_template('result.html', price=price)
	except Exception as e:
		return render_template('error.html', message=str(e))


if __name__ == '__main__':
	# Development server
	app.run(host='0.0.0.0', port=5000, debug=True)
