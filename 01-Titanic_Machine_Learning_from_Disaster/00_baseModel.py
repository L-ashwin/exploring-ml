import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn. model_selection import cross_val_score

# Reading the dataset
train = pd.read_csv('./dataset/train.csv', index_col=0)
test  = pd.read_csv('./dataset/test.csv',  index_col=0)

# Features 'Name' and 'Ticket' have specific values for each example(passenger)
# We will need to do some feature engineering to utlise them. Also 'Cabin' has
# lot of missing values and will need special attension as well.
# So, Dropping these features for the base model
features = ['Name', 'Ticket', 'Cabin']
train = train.drop(features, axis=1)
test  = test.drop(features, axis=1)

# dropping the examples with missing values for any of the features
train = train.dropna()

# splitting into features (xTrain) and labels (yTrain)
xTrain = train.drop('Survived', axis=1)
yTrain = train['Survived']
xTest  = test

#Encoding categorical features values to integers
catg_map = {}
for catg in ['Sex', 'Embarked']:
    unq = xTrain[catg].unique()
    catg_map[catg] = {key:val for val, key in enumerate(unq)}

    xTrain[catg] = xTrain[catg].map(catg_map[catg])
    xTest[catg]  = xTest[catg].map(catg_map[catg])

# instantiating RandomForestClassifier
estimator = LogisticRegression(tol=1e-4, solver='liblinear', random_state=1)

# Computing the cross validation accuracy as base model performance estimate.
cv = cross_val_score(estimator, xTrain, yTrain, cv=10)
print('Cross Validation Accuracy for the Base Model:', round(np.mean(cv), 4))
