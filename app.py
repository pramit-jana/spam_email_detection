import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
from urllib.parse import urlparse
import re
import requests
import tldextract
import whois
import socket
import numpy as np


# Initialize Flask application
app = Flask(__name__)
CORS(app)


## DATA PROCESS METHODS

# nltk.download('all')
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



    


### URL feature extract

# Function to check if the URL contains an IP address
def has_ip_address(url):
    try:
        hostname = urlparse(url).hostname
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        return 1 if re.match(ip_pattern, hostname) else 0
    except:
        return 0

# Function to check if URL is long
def is_long_url(url):
    return 1 if len(url) > 75 else 0

# Function to check if URL is shortened
def is_short_url(url):
    shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly']
    return 1 if any(shortener in url for shortener in shorteners) else 0
    

# Function to check if URL contains '@' symbol
def has_symbol_at(url):
    return 1 if '@' in url else 0


# Function to check if URL redirects with '//'
def is_redirecting(url):
    return 1 if '//' in url[url.find('://') + 3:] else 0



# Function to check if URL has '-' in domain
def has_prefix_suffix(url):
    return 1 if '-' in urlparse(url).netloc else 0


# Function to count subdomains
def count_subdomains(url):
    return len(tldextract.extract(url).subdomain.split('.'))



# Function to check if URL uses HTTPS
def uses_https(url):
    return 1 if urlparse(url).scheme == 'https' else 0




# Function to calculate domain registration length (requires WHOIS data)
def domain_registration_length(url):
    try:
        domain = whois.whois(urlparse(url).netloc)
        creation_date = domain.creation_date
        expiration_date = domain.expiration_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        return (expiration_date - creation_date).days if creation_date and expiration_date else 0
    except:
        return 0


# Function to check if a URL has a query string
def has_query_string(url):
    # Convert boolean True/False to integer 1/0
    return int(bool(urlparse(url).query))


# Function to check if the URL has a suspicious top-level domain (TLD)
def has_suspicious_tld(url):
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq']  # Example of known suspicious TLDs
    return 1 if any(url.endswith(tld) for tld in suspicious_tlds) else 0


# Function to check if the URL has a suspicious port number
def has_suspicious_port(url):
    return 1 if bool(re.match(r':\d{1,4}$', urlparse(url).netloc)) else 0



# Function to check if the URL contains numbers
def has_numbers(url):
    return int(bool(re.search(r'\d', url)))




# Function to check if the URL contains an uncommon protocol
def has_uncommon_protocol(url):
    uncommon_protocols = ['ftp', 'ssh', 'telnet']  # Example of uncommon protocols
    return 1 if any(url.lower().startswith(protocol) for protocol in uncommon_protocols) else 0


#Function to check suspicious keywords
def has_suspicious_keyword(url):
    return 1 if any(substring in url for substring in ['login', 'secure', 'bank', 'update', 'confirm']) else 0


def extract_features(url):
    features = np.array([has_ip_address(url), is_long_url(url), is_short_url(url),
                         has_symbol_at(url),is_redirecting(url),has_prefix_suffix(url),
                         count_subdomains(url),uses_https(url),domain_registration_length(url),
                         has_query_string(url),has_suspicious_tld(url),has_suspicious_port(url),
                         has_numbers(url),has_uncommon_protocol(url),has_suspicious_keyword(url)])
    return features




url_model = joblib.load("url_analyzer.pkl")



def extract_url(message):
 
    url_pattern = re.compile(
        r'http[s]?://'       
        r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|'   
        r'(?:%[0-9a-fA-F][0-9a-fA-F]))+',             
        re.IGNORECASE
    )
    
    # Search for the pattern in the message
    match = re.search(url_pattern, message)
    print(match)
    res=0
    if match:
        url=match.group(0)
        features = extract_features(url)
        features_2d = features.reshape(1, -1)
        predict = url_model.predict(features_2d)
        return 1 if predict==1 else 0
    else:
        return 0



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
    res=""
    print("Received request...")
    # Get input data from request
    data = request.get_json()
    print("Request data:", data)
    email = data['email']
    
    url_res=extract_url(email)
        
    if url_res==1:
        res="spam"
   
    else:
        prediction = predict_new_email(email)

        print("Sending response...")

        if prediction==0:
            res="ham"
        else:
            res="spam"
            
    return jsonify({'prediction': res})

if __name__ == '__main__':
    app.run(debug=False,port=8080)
