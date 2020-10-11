import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn. model_selection import cross_val_score

# Reading the dataset
train = pd.read_csv('./dataset/train.csv', index_col=0)
test  = pd.read_csv('./dataset/test.csv',  index_col=0)


# dropping the examples with missing values for any of the features
train = train.dropna()

# splitting into features (xTrain) and labels (yTrain)
xTrain = train.drop('Loan_Status', axis=1)
yTrain = train['Loan_Status']
xTest  = test

#Encoding categorical features values to integers
catg_map = {}
for catg in ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']:
    unq = xTrain[catg].unique()
    catg_map[catg] = {key:val for val, key in enumerate(unq)}

    xTrain[catg] = xTrain[catg].map(catg_map[catg])
    xTest[catg]  = xTest[catg].map(catg_map[catg])

# instantiating LogisticRegression
estimator = LogisticRegression(tol=1e-4, solver='liblinear', random_state=1)

# Computing the cross validation accuracy as base model performance estimate.
cv = cross_val_score(estimator, xTrain, yTrain, cv=10)
print('Cross Validation Accuracy for the Base Model:', round(np.mean(cv), 4))
