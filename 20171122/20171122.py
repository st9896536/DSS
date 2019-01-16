#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:44:54 2017

@author: Rong
"""

import numpy as np
npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(20), random_state=1)

X = np.append(trainface,trainnonface,axis = 0) #把兩個array上下接在一起
y = np.append(np.ones((trainface.shape[0],1)),np.zeros((trainnonface.shape[0],1)),axis = 0)
#X_new = (X - np.mean(X)) / np.std(X)
clf.fit(X,y)
y2 = clf.predict(X)
print(np.sum(y[:,0]==y2)/y.shape[0])

TX = np.append(testface,testnonface,axis = 0)
Ty = np.append(np.ones((testface.shape[0],1)),np.zeros((testnonface.shape[0],1)),axis = 0)

Ty2 = clf.predict(TX)
print(np.sum(Ty[:,0]==Ty2)/Ty.shape[0])
#1.減mean除std
#加幾個特徵去看accuracy會多少