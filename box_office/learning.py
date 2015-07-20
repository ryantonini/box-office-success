#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bootstrap aggregated neural network algorithm.

Created on Thu Jul  2 00:56:53 2015
@author: ryantonini
"""
from __future__ import division
import random
import copy

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError


class BaggingFNN():
    """bootstrap aggregated feed-forward neural network classifier"""
    
    
    def __init__(self, fnn, num_models=11):
        """create instance of BaggingFNN"""
       
        self.fnn = fnn 
        self.m = num_models # number of models to run bagging on 
        
    def bagging_classifier(self, trainInstances, testInstances, L):
        """Train and test bagging classifier for the neural network.  
            (1) generate self.m new training sets each with L instances 
            from trainInstances using replacement;
            (2) train self.m neural networks on each of the self.m training sets; 
            (3) majority vote
        
        Precondition: dimensions of trainInstances,testInstances must match self.fnn
        
        :param trainInstances: collection of training examples
        :type trainInstances: ClassificationDataSet
        :param testInstances: collection of test examples
        :type testInstances: ClassificationDataSet
        :param L: number of items in each training set
        :type L: int
        :returns: accuracy of predictions
        :rtype: float
        """ 
        ensemble = []
        for j in range(self.m):
            # generate random sample of indices
            tset = random.sample(range(0, len(trainInstances["input"])), L) 
            c = ClassificationDataSet(self.fnn.indim, 1, nb_classes=self.fnn.outdim)
            for index in tset:
                c.appendLinked(trainInstances['input'][index], trainInstances['target'][index])
            c._convertToOneOfMany(bounds=[0,1]) # 1 of k binary representation
            net = buildNetwork(24, 18, 16, 8, hiddenclass=TanhLayer, outclass=SoftmaxLayer) # define neural net
            trainer = BackpropTrainer(net, dataset=c, learningrate=0.01, momentum=0.1, verbose=True, weightdecay=0.01)
            trainer.trainEpochs(20) # train
            ensemble.append(net)
            print percentError(trainer.testOnClassData(
                                dataset=testInstances ), testInstances['class'])
        # key is test example, value is list of labels from each model    
        d = dict.fromkeys(np.arange(len(testInstances['input']))) 
        for model in ensemble:
            # get label with highest probability for each test example
            result = model.activateOnDataset(testInstances).argmax(axis=1)
            for k in range(len(result)):
                if d[k] == None:
                    d[k] = [result[k]]
                else:
                    d[k].append(result[k])
        predictions = []
        for ex in d.keys():
            predictions.append(max(set(d[ex]), key=d[ex].count)) # majority voting 
        actual = [int(row[0]) for row in testInstances['class']]
        return accuracy_score(actual, predictions) # traditional accuracy calc
        
    def cross_validate(self, data, folds=9):
        """Apply stratified k-fold cross validation.
        
        Precondition: dimensions of data must match self.fnn
        
        :param data: table of features, labels for each training example
        :type data: dataFrame
        :param folds: number of folds (must be at least 2)
        :type folds: int
        :return: true accuracy 
        :rtype: float
        
        """
        total = 0
        features = np.array(data[data.columns[:-1]])
        targets = np.array(data['box office'])
        skf = StratifiedKFold(targets, n_folds=folds) # stratified k-fold cv
        for train_index, test_index in skf:  # iterate each fold
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = targets[train_index], targets[test_index]
            ds_train = self.generate_dataset(X_train, y_train) # training dataset
            ds_test = self.generate_dataset(X_test, y_test) # testing dataset
            ds_test._convertToOneOfMany(bounds=[0,1]) # 1 of k binary representation
            s = self.bagging_classifier(ds_train, ds_test, 
                                                  len(ds_train['input']))
            print s
            total+=s
        return total/folds
    
    def generate_dataset(self, X, y):
        """Generate a classification dataset based on features X and targets y"""
        ds = ClassificationDataSet(self.fnn.indim, 1, nb_classes=self.fnn.outdim) 
        for j in range(len(y)):
            ds.appendLinked(X[j], y[j]-1)
        return ds
    
           
def fnn_accuracy(trainer, train_data, test_data, epochNum=100):
    
    trainer.trainUntilConvergence(dataset=train_data, maxEpochs=epochNum)
    train_result = percentError(trainer.testOnClassData(),
                              train_data['class'])
    test_result = percentError(trainer.testOnClassData(
                    dataset=test_data), test_data['class'])
    print "Train accuracy: %5.2f%%" % (100-train_result)
    print "Test accuracy: %5.2f%%" % (100-test_result)

    return (100-train_result, 100-test_result)


    
if __name__ == "__main__":
    
    df = pd.read_csv("movie_dataset.csv")
    net = buildNetwork(24, 20, 16, 8, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
    bag_fnn = BaggingFNN(net)
    result = bag_fnn.cross_validate(df)
    print "Accuracy Rating: ", result  
    #for 8 folds, 0.29772



