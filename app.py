from flask import Flask, request, jsonify
from data_process_method import data_process
from flask_cors import CORS
import joblib

# Initialize Flask application
app = Flask(__name__)
CORS(app)
model = joblib.load('model.pkl')


import string
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def data_process(t):
    t=t.lower()
    t=nltk.word_tokenize(t)
    l=[]
    for i in t:
        if i.isalnum():
            l.append(i)
    t=list(l)
    l.clear()
    for i in t:
        if i not in stopwords.words('english') and i not in string.punctuation:
            l.append(i)
    t=list(l)
    l.clear()
    for i in t:
        l.append(ps.stem(i))
    
    return " ".join(l)

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
def hello():
    return "Hello"

if __name__ == '__main__':
    app.run(debug=True, port='0.0.0.0') 
