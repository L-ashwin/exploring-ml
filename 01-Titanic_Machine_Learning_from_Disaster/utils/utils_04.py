import re
import json
import numpy as np
import pandas as pd

from sklearn. model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def group_titles(titles):
    for i, each in enumerate(titles):
        if  any(each == ele for ele in ['Mr.', 'Miss.', 'Mrs.', 'Master.']):
            continue
        elif  any(each == ele for ele in ['Sir.', 'Ms.', 'Mme.', 'Mlle.', 'Lady.', 'Countess.']):
            titles[i] = 'grp1'
        else:
            titles[i] = 'grp2'

def parameterTune(estimator, param_grid):
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(
            estimator  = estimator,
            param_grid = param_grid,
            n_jobs     = 11,
            cv         = 5,

    )
    grid.fit(xTrain, yTrain)

    return grid.best_score_, grid.best_params_

# function to generate submission file
def test_eval(estimator, params):
    clf = estimator(**params)
    clf.fit(xTrain, yTrain)
    yPred = clf.predict(xTest)

    df = pd.DataFrame({'PassengerId':Test_id, 'Survived':yPred})
    return df



# Reading the dataset
Train = pd.read_csv('./dataset/train.csv', index_col=0)
Test  = pd.read_csv('./dataset/test.csv',  index_col=0)

# Pre-Processing
features = ['Ticket', 'Cabin']
Train = Train.drop(features, axis=1)
Test  = Test.drop(features, axis=1)
Test_id = Test.index

Train['Name'] = Train.Name.map(lambda x:re.findall('([A-Za-z]+\.)' ,x)[0])
Test['Name']  = Test.Name.map(lambda x:re.findall('([A-Za-z]+\.)' ,x)[0])

group_titles(Train.Name.values)
group_titles(Test.Name.values)

for attr in ['Age']: #fillna for real valued features with mean
    fill = Train[attr].mean()
    Train[attr].fillna(fill, inplace=True)
    Test[attr].fillna(fill, inplace=True)

# as Fare has skewed distribution using median as central tendancy
for attr in ['Fare']: #fillna for real valued features with median
    fill = Train[attr].median()
    Train[attr].fillna(fill, inplace=True)
    Test[attr].fillna(fill, inplace=True)

for attr in ['Embarked']: #fillna for categorical features with mode
    fill = Train[attr].mode()[0]
    Train[attr].fillna(fill, inplace=True)
    Test[attr].fillna(fill, inplace=True)

train = pd.get_dummies(Train)
test  = pd.get_dummies(Test)

# splitting into features (xTrain) and labels (yTrain)
xTrain = train.drop('Survived', axis=1)
yTrain = train['Survived']
xTest  = test

scaller = StandardScaler()
scaller.fit(xTrain[['Age', 'Fare']])

xTrain[['Age', 'Fare']] = scaller.transform(xTrain[['Age', 'Fare']])
xTest[['Age', 'Fare']]  = scaller.transform(xTest[['Age', 'Fare']])
