from flask import Flask, request, jsonify
from data_process_method import data_process
import joblib

# Initialize Flask application
app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/predict/<text>', methods=['GET'])
def predict(text):
    print("Received request...")

    # Get input data from URL parameter
    input_text = text

    # Perform data preprocessing
    processed_text = data_process(input_text)

    prediction = model.predict([processed_text])[0]

    # Return the prediction
    print("Sending response...")
    if prediction == 0:
        res = "ham"
    else:
        res = "spam"

    return jsonify({'prediction': res})
    
@app.route('/hello', methods=['GET'])
def predict():
    return "Hello"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 