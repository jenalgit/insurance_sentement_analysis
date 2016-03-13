import pandas as pd
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
    {"tweets" :[
    {"text":"best company ever"},
    {"text":"this company sucks"},
    {"text":"you guys are great"}
    ]}
    """

    # get data, xform to a pandas series
    data = request.get_json(force=True)
    tweets = pd.Series( [item["text"] for item in data['tweets'] ])

    # make score predictions
    tweets = tweets.map(clean_tweets)
    X = vectorizer.transform(tweets)
    results = clf.predict_proba(X)[:,1]

    # create JSON of results
    tweet_index = range(1, len(results)+1)
    scores = dict(zip(tweet_index, results))

    # return the json
    return jsonify(results=scores)
    
if __name__ == '__main__':
    app.run(port=7000, debug=True)
    
    