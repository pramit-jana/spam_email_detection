from flask import Flask, request, jsonify
from data_process_method import data_process
import joblib

# Initialize Flask application
app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request...")
    # Get input data from request
    data = request.get_json()
    print("Request data:", data)
    email = data['email']
    
    # Perform data preprocessing
    input_email = data_process(email)

    prediction = model.predict([input_email])[0]

    # Return the prediction
    print("Sending response...")

    if prediction==0:
        res="ham"
    else:
        res="spam"

    return jsonify({'prediction': res})

if __name__ == '__main__':
    app.run(debug=True,port=8080)  # You can set debug to False in production
