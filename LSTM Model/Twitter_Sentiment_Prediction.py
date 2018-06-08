import tweepy
import keys
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

consumer_key = keys.ConsumerKey
consumer_secret = keys.ConsumerSecret
access_token = keys.AccessToken
access_token_secret = keys.AccessTokenSecret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
public_tweets = api.search('life')

model = load_model(filepath='model.h5')

def sentiment_analysis(text):
    tokenizer = Tokenizer(num_words=10, split= ' ')
    tokenizer.fit_on_texts(text)
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text, maxlen=29, dtype='int32', value=0)
    sentiment = model.predict(text,batch_size=1,verbose = 2)[0]
    if(np.argmax(sentiment) == 0):
        print("negative")
    elif (np.argmax(sentiment) == 1):
        print("positive")

for tweet in public_tweets:
    sentiment_analysis(tweet.text)
    print(tweet.text)
