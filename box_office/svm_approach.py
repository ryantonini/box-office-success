# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 22:11:06 2015

@author: ryantonini
"""
import pandas as pd
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score


def svmClassifier(x_train, x_test, y_train, y_test):
    """Determine the accuracy score of the SVM model on the test data.
    
    This function uses the SVC class to perform multiclass classification on 
    the dataset.  The class is defined within the Scikit-Learn library.
    
    :param x_train: training data (input)
    :param x_test: test data (input)
    :param y_train: training data (output)
    :param y_test: test_data (output)
    :return: accuracy score
    """
    clf = svm.SVC(kernel='rbf') 
    clf.fit(x_train, y_train) 
    pred = clf.predict(x_test)
    return accuracy_score(y_test, pred)

def cross_validate(x, y, folds=11):
    """Apply stratified k-fold cross validation.
    
    :param x: samples
    :param y: class labels for samples
    :param folds: number of folds (must be at least 2)
    :return: true accuracy 
    """
    total = 0
    skf = StratifiedKFold(y, n_folds=folds) # stratified k-fold cv
    for train_index, test_index in skf:  # iterate each fold
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        score = svmClassifier(X_train, X_test, y_train, y_test)
        total += score
    return total/folds
    
    
if __name__ == "__main__":
    
    df = pd.read_csv("movie_dataset.csv")
    y = np.array(df['box office'])
    x = np.array(df.loc[:, "sequel": "thriller"])
    score = cross_validate(x, y)
    print "Prediction Accuracy: ", score
    