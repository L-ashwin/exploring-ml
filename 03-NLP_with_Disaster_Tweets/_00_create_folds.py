#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:08:20 2020

@author: ashwin
"""
import numpy as np
import pandas as pd

def add_folds(dataframe, y, n_folds):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=22)
    
    fold = np.zeros(dataframe.shape[0])
    for i, (train_index, test_index) in enumerate(skf.split(dataframe, y)):
        fold[test_index] = i
    
    dataframe['fold'] = fold
    
    return dataframe


if __name__ == '__main__':
    # load dataset
    data = pd.read_csv('./dataset/train.csv', index_col=0)
    
    # adding folds
    data = add_folds(data, data.target, 5)
    
    # saving dataframe with folds
    data.to_csv('./dataset/train_folds.csv', index=None)