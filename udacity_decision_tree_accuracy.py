# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:09:47 2018

@author: MLUSER
"""

import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn import tree

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

#################################################################################
########################## DECISION TREE #################################
#### your code goes here

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
y_label = clf.predict(features_test)

acc = np.sum((y_label == labels_test).astype(int)) / float(len(labels_test))
### you fill this in!
### be sure to compute the accuracy on the test set
    
def submitAccuracies():
  return {"acc":round(acc,3)}