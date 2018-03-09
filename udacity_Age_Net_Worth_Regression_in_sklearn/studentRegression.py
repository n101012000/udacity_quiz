# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 12:47:45 2018

@author: MLUSER
"""

from sklearn import linear_model

def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    ### name your regression reg
    
    ### your code goes here!
    
    reg = linear_model.LinearRegression()
    reg.fit(ages_train, net_worths_train)
    
    return reg