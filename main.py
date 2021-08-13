# -------------------------------------------------------------------------------
# Name: main.py
# Purpose: Pull data from twitter, perform sentiment analysis and pull stock data
#
# Author(s):    David Little
#
# Created:      04/26/2021
# Updated:
# Update Comment(s):
#
# TO DO:
#
# -------------------------------------------------------------------------------

import requests
import pandas as pd
import time
import regex as re
from datetime import datetime, timedelta



def get_data(tweet):
    data = {
        'id': tweet['id'],
        'created_at': tweet['created_at'],
        'text': tweet['text'],
#        'retweet_count': tweet['public_metrics']['retweet_count'],
#        'like_count': tweet['public_metrics']['like_count'],
#        'reply_count': tweet['public_metrics']['reply_count']
#        'quote_count': tweet['public_metrics']['quote_count']
    }
    return data

whitespace = re.compile(r"\s+")
web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
tesla = re.compile(r"(?i)@Tesla(?=\b)")
user = re.compile(r"(?i)@[a-z0-9_]+")

#------------------------------------- Twitter Pull  --------------------------------------------------------

# setup the API request
endpoint = 'https://api.twitter.com/1.1/search/tweets.json' #'https://api.twitter.com/2/tweets/search/recent'  # 'https://api.twitter.com/2/tweets/search/all'
headers = {'authorization': f'Bearer {BEARER_TOKEN}'}
params = {
    'q': '(tesla OR tsla OR elon musk) (lang:en) -is:retweet',
    'count': '100',
#    'tweet.fields': 'created_at,lang,public_metrics',
    'result_type': 'popular'
        }

dtformat = '%a %b %d %H:%M:%S +0000 %Y'#'%Y-%m-%dT%H:%M:%SZ'  # the date format string required by twitter

# we use this function to subtract 60 mins from our datetime string
def time_travel(now, d):
    now = datetime.strptime(now, dtformat)
    back_in_time = now - timedelta(days=d)
    return back_in_time.strftime(dtformat)

now = datetime.now()  # get the current datetime, this is our starting point
last_week = now - timedelta(days=6)  # datetime one week ago = the finish line
now = now.strftime(dtformat)  # convert now datetime to format for API
now

df = pd.DataFrame()  # initialize dataframe to store tweets
while True:
    if datetime.strptime(now, dtformat) < last_week:
        # if we have reached 6 days ago, break the loop
        break
    pre60 = time_travel(now, 1)  # get x minutes before 'now'
    # assign from and to datetime parameters for the API
    #params['start_time'] = pre60
    params['until'] = (datetime.strptime(now, dtformat)).strftime('%Y-%m-%d')
    response = requests.get(endpoint,
                            params=params,
                            headers=headers)  # send the request
   # time.sleep(2)
    now = pre60  # move the window 60 minutes earlier
    # iteratively append our tweet data to our dataframe
    for tweet in response.json()['statuses']:
        row = get_data(tweet)  # we defined this function earlier
       # if row['like_count']>=1 and row['retweet_count']>=1 and row['reply_count']>=0:   #row['like_count'] >=3:
        df = df.append(row, ignore_index=True)
#df
df = df.drop_duplicates()
#---------------------------------------------- Sentiment Model ------------------------------------------------------

import flair
sentiment_model = flair.models.TextClassifier.load('en-sentiment')

# we will append probability and sentiment preds later
probs = []
sentiments = []
clean_tweets = []
timestamp = []
binary = []

for time in df['created_at']:
    timestamp.append(((datetime.strptime(time, dtformat)
                      - timedelta(hours = 4)) #timezone
                      - timedelta(hours = 0) #delay
                     ).strftime('%Y-%m-%d'))  # %H:00:00'))

for tweet in df['text']:
# we then use the sub method to replace anything matching
    tweet = whitespace.sub(' ', tweet)
    tweet = web_address.sub('', tweet)
    tweet = tesla.sub('Tesla', tweet)
    tweet = user.sub('', tweet)
    sentence = flair.data.Sentence(tweet)
    sentiment_model.predict(sentence)
    # extract sentiment prediction
    sentiments.append(sentence.labels[0].value)  # 'POSITIVE' or 'NEGATIVE'
    if sentence.labels[0].value == 'NEGATIVE':
        probs.append(-1 * sentence.labels[0].score)  # numerical score 0-1
        binary.append(0)
    else:
        probs.append(sentence.labels[0].score)  # numerical score 0-1
        binary.append(1)
    clean_tweets.append(tweet)
    # print(tweet)
    # print(' ')

# add probability and sentiment predictions to tweets dataframe
df['text_clean'] = clean_tweets
df['probability'] = probs
df['sentiment'] = sentiments
df['binary'] = binary
df['Date'] = timestamp
#df['Date'] = pd.to_datetime(df['Date'])
df

#________________________________ Stock Data __________________________________________________________________

import yfinance as yf

tsla = yf.Ticker("TSLA")
tsla_stock = tsla.history(
    start=df['Date'].min(),
    end=df['Date'].max(),
    interval='1d'   #'60m'
        ).reset_index()
tsla_stock

converted = []
for time in tsla_stock['Date']:
    converted.append(time.strftime('%Y-%m-%d'))  # %H:00:00'))
tsla_stock['Date'] = converted
#tsla_stock['Date'] = pd.to_datetime(tsla_stock['Date'])
tsla_stock

means = df.groupby(['Date'],  as_index=False).mean()
means

combined = means.merge(tsla_stock, how='inner')
combined


#combined['like_count'].corr(combined['Close'])
print(combined['binary'].corr(combined['Close']))


import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1)

# plot the data
ax.plot(combined['binary'],combined['Close'], 'ro')


#import tweepy

#auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
#auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

#api = tweepy.API(auth)

#public_tweets = api.home_timeline()
#for tweet in public_tweets:
#    print(tweet.text)

