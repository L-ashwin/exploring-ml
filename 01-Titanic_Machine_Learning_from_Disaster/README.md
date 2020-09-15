# Titanic: Machine Learning from Disaster
This classification task is taken from the Kaggle Competition, [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic).  
I will be using the strategy explained by [Andrew Ng](https://twitter.com/AndrewYNg) in his [ML advice lecture](https://bit.ly/3kqyiKB).
I will first create the simple base model and get the score/performance, and then observe the change in performance as I go on increasing the complexity.

## 00_baseModel.py
This file contains the base model for the classification task. 
No feature engineering was done, examples with NA values are dropped.
Logistic Regression Classifier was used for classification and cross-validation accuracy as a performance measure.  
With these settings, the Cross-Validation accuracy of 78.66 was achieved.

## 01_exploratory_data_analysis.ipynb
This file contains the exploratory data analysis for the Titanic data set.
Insights from this will later be used to improve our classification model.
