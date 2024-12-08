import numpy as np
from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)
model = load('fraud_detection_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if a given input is fraudulent or not.

    Returns
    -------
    JSON
        A JSON object containing the prediction result:
        - 1 indicates a fraudulent transaction.
        - 0 indicates a non-fraudulent transaction.
    """
    test_data = np.array(request.json).reshape(1, -1)
    prediction = 1 if model.predict(test_data) == -1 else 0
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
