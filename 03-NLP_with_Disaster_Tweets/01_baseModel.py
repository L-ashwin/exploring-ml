import re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

def preProc(tweet, stemmer):
    tweet = re.sub('[^A-Za-z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [stemmer.stem(word) for word in tweet if word not in stopwords.words('english')]
    tweet = ' '.join(tweet)
    return tweet

# load dataset
data = pd.read_csv('./dataset/train.csv', index_col=0)

# pre-processing tweets
stemmer = PorterStemmer()
corpus = [preProc(tweet, stemmer) for tweet in data.text]

# CountVectorizer feature extractor
fe = CountVectorizer(max_features=5000, ngram_range=(1,3))
fe.fit(corpus)

# transform tweets using CountVectorizer feature extractor
xData = fe.transform(corpus).toarray()
yData = data.target

# Instantiate the classifier
clf = MultinomialNB()
score = np.mean(cross_val_score(clf, xData, yData))

print('Accuracy for base model:', round(score,2))
