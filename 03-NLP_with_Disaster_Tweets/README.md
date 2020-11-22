# NLP with Disaster Tweets
This classification task is from the Kaggle Competition, [NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started).  
I will be first exploring methods to extract features from the text, and analyze their performance on this classification task. 
Starting with simple statistical methods will move on to the latest deep learning techniques for feature extraction and classification.

## 00 Utility Functions

### create_folds
While tuning parameters for the model or comparing between different models it is advisable to evaluate them on equal ground i.e on the same validation and training set. This script is used to split the data into different folds and write these folds to CSV. An extra column is added to the original data frame called folds which has fold number in it. Each of those different folds will be used to test the model training on the remaining folds.

### runBuilder
This function is used to create different combinations of parameters while parameter tuning. 

## 01 Base Model

### baseModel
The base model for this text classification task is built with Count Vectorizer as a feature extraction technique and Multinomial Nieve Bayes as a classifier.

### baseModel_2
After building a simple base model, folds were created and saved. This script is used to evaluate the cross-validation score for the base model.

## 02 vectorizer_tuning
In this notebook, a different way of extracting features from the text (TFIDF) is explored. Cross-validation scores for both count-vectorizer and TFIDF are computed with different parameter combinations.

## 03 gru-lstm
The use of word embeddings is one of the elegant ways of transfer learning in NLP where we get to pass in information to the model from outside the dataset. [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) is one of the famous algorithms used to generate embeddings. GloVe vectors trained on huge corpus are made available by Stanford [here](https://nlp.stanford.edu/projects/glove/)  
In this notebook, GloVe embeddings are used to extract features from the text. After enoding each word with an embedding vector, we get a sequence of such vectors for each tweet. RNN's are an effective way to deal with such sequential data. A couple of modified RNNs (LSTM & GRU) are explored in this notebook for classification.














