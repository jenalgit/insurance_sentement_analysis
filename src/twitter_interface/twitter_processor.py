# tweepy and auth are for twitter.  Don't check in auth
import tweepy
import auth

import os
import pandas as pd
import datetime
import time
import numpy as np

# needed for plotly visualization
from plotly.offline import plot
import plotly.graph_objs as go

# sending requests
import requests
import json

# global hook to twitter
auth_handle = tweepy.OAuthHandler(auth.consumer_key, auth.consumer_secret)
auth_handle.set_access_token(auth.access_token, auth.access_token_secret)
api = tweepy.API(auth_handle)

def get_tweets(handle_list, tweet_count):
    """
    returns top x tweets for all handles in a list
    handle_list = list of handles
    tweet_count = how many tweets to get
    """
    results = {}
    for handle in handle_list:
        results[handle] = ["\"" + item.text + "\"" for item in api.search(to=handle, count=tweet_count)]
    return results

def store_scores(scores):
    """
    This method will take the timestamped scores from the classifier and store/manage them in a dataframe
    """
    # if a score file exists, append to it
    if os.path.isfile("scores.csv"):
        score_df = pd.read_csv("scores.csv", index_col='time')
        # create a df, using only the new score and append to the existing df
        new_score_df = pd.DataFrame([scores])
        new_score_df = new_score_df.set_index('time')
        score_df = score_df.append(new_score_df)
        # then save
        # print (score_df)
        score_df.to_csv("scores.csv")
    else:
        # otherwise we'll just create a new df with the current entry
        score_df = pd.DataFrame([scores])
        score_df = score_df.set_index('time')
        score_df.to_csv("scores.csv")

    return score_df

def classify_bulk_tweets(tweet_dict):


    scores = {}

    # timestamp for the score, will become a dataframe index.
    score_time = datetime.datetime.now()
    scores['time'] = str(score_time)

    for key in tweet_dict.keys():

        tweets = tweet_dict[key]

        # format our json message
        tweet_list = [{"text": item} for item in tweets]
        json_data = {"tweets":tweet_list}
        # send request
        r = requests.post("http://127.0.0.1:7000/api", json=json_data)
        classifier_results = r.json()
        # compute the mean value
        mean_value = np.array([value for value in classifier_results['results'].values()]).mean()
        scores[key] = mean_value
    return scores

def update_graph(scores_df):
    data = [go.Scatter(x=scores_df.index, y=scores_df['allstate'], name="Allstate"),
            go.Scatter(x=scores_df.index, y=scores_df['geico'], name="Geico"),
            go.Scatter(x=scores_df.index, y=scores_df['progressive'], name="Progressive"),
            go.Scatter(x=scores_df.index, y=scores_df['statefarm'], name="State Farm"),
            go.Scatter(x=scores_df.index, y=scores_df['usaa'], name="USAA"),
            go.Scatter(x=scores_df.index, y=scores_df['wearefarmers'], name="Farmers"),
            ]

    layout = go.Layout(title="Customer Sentiment Towards Top 6 Auto Insurance Companies",
                   xaxis=dict(title="Time"),
                   yaxis=dict(title="Customer Sentement")
              )

    fig = go.Figure(data=data, layout=layout)

    plot(fig, link_text="", filename="plot.html", auto_open=False)

def runLoop():
    top6 = ['statefarm', 'geico', 'allstate', 'progressive', 'wearefarmers', 'usaa' ]
    # get twitter data from top 6
    results = get_tweets(top6, 10)

    # classify
    scores = classify_bulk_tweets(results)
    # store scores and get dataframe
    scores_df = store_scores(scores)
    # graph scores
    update_graph(scores_df)

def main():

    while True:
        try:
            runLoop()
            time.sleep(60)
        except KeyboardInterrupt:
            print("Manual Break by User")
            return


if __name__ == "__main__": main()

