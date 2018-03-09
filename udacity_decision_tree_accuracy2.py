# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:22:39 2018

@author: MLUSER
"""

import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn import tree

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



########################## DECISION TREE #################################


### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively


clf_min_samples_split_2 = tree.DecisionTreeClassifier(min_samples_split=2)
clf_min_samples_split_2 = clf_min_samples_split_2.fit(features_train, labels_train)
y_label_min_samples_split_2 = clf_min_samples_split_2.predict(features_test)

acc_min_samples_split_2 = np.sum((y_label_min_samples_split_2 == labels_test).astype(int)) / float(len(labels_test))

clf_min_samples_split_50 = tree.DecisionTreeClassifier(min_samples_split=50)
clf_min_samples_split_50 = clf_min_samples_split_50.fit(features_train, labels_train)
y_label_min_samples_split_50 = clf_min_samples_split_50.predict(features_test)

acc_min_samples_split_50 = np.sum((y_label_min_samples_split_50 == labels_test).astype(int)) / float(len(labels_test))


def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}