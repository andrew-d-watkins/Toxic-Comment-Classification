# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

eng_stopwords = set(stopwords.words("english"))
lem = WordNetLemmatizer()

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if toxic_model:
        try:
            req_data = request.get_json()
            query = pd.DataFrame(req_data['comments'])
            query.iloc[:,0] = query.iloc[:,0].apply(lambda x: clean_comment(x))
        
            return jsonify({'prediction': str(toxic_model.predict(query.iloc[:,0]))})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
    
@app.route('/clean', methods=['POST'])    
def clean_comment(text):
    
    #split into words
    tokens = word_tokenize(text)
    #change words to lower case and lemmatize
    stemmed = [lem.lemmatize(word.lower()) for word in tokens]
    #remove stop words
    words = [w for w in stemmed if not w in eng_stopwords]
    #remove anything non-alphabetic
    clean_words = [word for word in words if word.isalpha()]
    #append to string
    clean_comment = " ".join(clean_words)
    
    return clean_comment

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    print('Model is loading...')
    #toxic_model = joblib.load("toxic_model.pkl") # Load "model.pkl"
    toxic_model = joblib.load("sample_model.pkl")
    print ('Model loaded')
    #model_columns = joblib.load("demo_model_columns.pkl") # Load "model_columns.pkl"
    #print ('Model columns loaded')

    app.run(port=port, debug=True)