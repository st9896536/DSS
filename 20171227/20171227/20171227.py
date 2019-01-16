#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:48:01 2017

@author: Rong
"""

import numpy as np
D = np.load('PCA_data.npy')

#降維

def PCATrain(D,R): #D原始資料 R是ratio
    M, F = D.shape  #M＝12943 * 50
    meanv = np.mean(D,axis=0)
    D2 = D - np.matlib.repmat(meanv,M,1) #1*50的矩陣repeat12943次
    C = np.dot(np.transpose(D2),D2) #C 50*50 coviarience metrix
    EValue, Evector = np.linalg.eig(C)
    EV2 = np.cumsum(EValue)/np.sum(EValue) 
    num = np.where(EV2>=R)[0][0]+1 
    return meanv, Evector[:,range(num)]
    

def PCATest(D,meanv,W): #W轉置矩陣50*7
    M, F = D.shape
    D2 = D - np.matlib.repmat(meanv,M,1)
    D3 = np.dot(D2,w)
    return D3 #M*7的矩陣

meanv, w = PCATrain(D,0.8)
newD = PCATest(D,meanv,w)