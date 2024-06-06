import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib


# Initialize Flask application
app = Flask(__name__)
CORS(app)


## DATA PROCESS METHODS

nltk.download('all')
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

def char_count(text):
    return len(text)


def word_count(text):
    return len(nltk.word_tokenize(text))

def count_suspicious_keywords(email_text):
    suspicious_keywords = [
        'offer', 'buy', 'cheap', 'click', 'subscribe', 'urgent', 
        'important', 'password', 'credit card', 'lottery', 'winner', 
        'claim', 'limited time', 'exclusive', 'act now','free','gift card','account','link'
        ,'balance'
    ]
    count = sum(keyword in email_text.lower() for keyword in suspicious_keywords)
    return count



### PREDICT 

X_train=pd.read_csv('X_train.csv')

def predict_new_email(new_email, model_path='model.pkl'):
    # Load the models and vectorizer
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Preprocess the email
    processed_email = data_process(new_email)
    email_vector = tfidf_vectorizer.transform([processed_email]).toarray()

    # Additional features
    count_char = char_count(processed_email)
    count_word = word_count(processed_email)
    suspicious_keyword = count_suspicious_keywords(new_email)
    
    basic_features_df = pd.DataFrame([{
        'char_count': count_char,
        'word_count': count_word,
        'suspicious_keywords': suspicious_keyword
    }])

    email_vector_df = pd.DataFrame(email_vector, columns=tfidf_vectorizer.get_feature_names_out())
    email_features = pd.concat([email_vector_df, basic_features_df], axis=1)
    email_features = email_features.reindex(columns=X_train.columns, fill_value=0)
    
    # Predict
    prediction = model.predict(email_features)
    return prediction[0]



@app.route('/predict', methods=['POST'])
def predict():
    print("Received request...")
    # Get input data from request
    data = request.get_json()
    print("Request data:", data)
    email = data['email']
    
   
    prediction = predict_new_email(email)

    # Return the prediction
    print("Sending response...")

    if prediction==0:
        res="ham"
    else:
        res="spam"

    return jsonify({'prediction': res})

if __name__ == '__main__':
    app.run(debug=False,port=8080)  # You can set debug to False in production
