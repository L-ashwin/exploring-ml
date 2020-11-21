import re
from nltk.corpus import stopwords

def preProc(tweet, remove_stopwords = False):
    tweet = re.sub('http\S+','', tweet)
    tweet = re.sub('@\S+','', tweet)
    tweet = re.sub('[^A-Za-z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    if remove_stopwords:
        tweet = [word for word in tweet if word not in stopwords.words('english')]
    tweet = ' '.join(tweet)
    return tweet