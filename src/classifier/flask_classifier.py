import pandas as pd
import numpy as np
from flask import Flask, abort, jsonify, request
from sklearn.externals import joblib
import re
from json import dumps

# load vectorizer and classifier
vectorizer = joblib.load("vectorizer.joblib")
clf = joblib.load("bayes_classifier.joblib")

def clean_tweets(x):
    x = x.lower()  #lowercase everything
    x = re.sub('@[a-z_-]+',"",x) #remove handles
    x = re.sub('[0-9]+',"",x)  #remove numbers
    x = re.sub('(https?):\/\/(www\.)?[a-z0-9\.:].*?(?=\s)', "",x) #remove urls
    x = re.sub('&[a-z]+','',x)  #remove html punctuations &amp, etc...
    return x

app = Flask(__name__)


@app.route('/api', methods=['POST'])
def make_predict():
    """
     JSON should look like this

    [{"company1" :[
    {"text":"best company ever"},
    {"text":"this company sucks"},
    {"text":"you guys are great"}],
    {"company2" :[
    {"text":"best company ever"},
    {"text":"this company sucks"},
    {"text":"you guys are great"}]
    ]
    """

    # get data, xform to a dict of  pandas series
    data = request.get_json(force=True)
    company_dict = {}
    results_dict = {}
    for key in data.keys():
        company_dict[key] = pd.Series([item["text"] for item in data[key] ])

    # make score predictions
    for key in company_dict.keys():

        try:
            company_dict[key] = company_dict[key].map(clean_tweets)
            X = vectorizer.transform(company_dict[key])
            results = clf.predict_proba(X)[:, 1]
        except:
            print("Unexpected error:", sys.exc_info()[0])
            results = np.zeros(len(company_dict[key]))

        # create JSON of results
        tweet_index = range(1, len(results)+1)
        scores = dict(zip(tweet_index, results))
        results_dict[key] = scores

    # return the json
    return jsonify(results=results_dict)
    
if __name__ == '__main__':
    app.run(port=7000, host='0.0.0.0', debug=True)
    
    