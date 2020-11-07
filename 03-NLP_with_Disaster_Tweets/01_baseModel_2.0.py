import re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

def preProc(tweet, stemmer):
    tweet = re.sub('[^A-Za-z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [stemmer.stem(word) for word in tweet if word not in stopwords.words('english')]
    tweet = ' '.join(tweet)
    return tweet

def get_cross_val_score(xData, yData, fold_series, fe, clf, metric):
    scores = []
    for fold in fold_series.unique():
        xTrain = xData[fold_series != fold]
        yTrain = yData[fold_series != fold]
        xTest  = xData[fold_series == fold]
        yTest  = yData[fold_series == fold]

        # train feature extractor on train data
        fe.fit(xTrain)
        
        # transform tweets using CountVectorizer feature extractor
        xTrain, xTest = fe.transform(xTrain), fe.transform(xTest)
        
        # train the classifier
        clf.fit(xTrain, yTrain)
        yPred = clf.predict(xTest)
        
        # performance metric
        score = metric(yTest, yPred)
        scores.append(score)
        
    return scores

if __name__ == '__main__':
    # load dataset
    data = pd.read_csv('./dataset/train_folds.csv', index_col=0)
    
    # x, y split 
    xData = data.text
    yData = data.target
    folds = data.fold
    
    # pre-processing tweets
    stemmer = PorterStemmer()
    xData = xData.map(lambda tweet:preProc(tweet, stemmer)) #[preProc(tweet, stemmer) for tweet in xData]
    
    # CountVectorizer feature extractor
    fe = CountVectorizer(max_features=100, ngram_range=(1,3))
    
    # MultinomialNB as classifier & accuracy_score as performance metric
    estimator = MultinomialNB()
    metric = accuracy_score
    
    # Compute cross validation score
    scores = get_cross_val_score(xData, yData, folds, fe, estimator, metric)
    
    print('cross val accuracy for base model:', round(np.mean(scores),2))