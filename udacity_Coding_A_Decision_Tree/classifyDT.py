# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:01:36 2018

@author: MLUSER
"""

from sklearn import tree

def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    clf = tree.DecisionTreeClassifier();
    clf = clf.fit(features_train, labels_train)
    
    return clf