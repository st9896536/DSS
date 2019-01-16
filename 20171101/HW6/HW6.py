#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 23:15:48 2017

@author: Rong
"""

#把19 * 19的照片拉成一維 共有2429張照片 把他load進資料庫
import os
from PIL import Image
import numpy as np
import random
import math
import matplotlib.pyplot as plt


"""
os.chdir('/Users/apple/Desktop/train/face')
filelist = os.listdir()
x = np.zeros((len(filelist),19*19))
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:] = np.array(IMG.getdata())
trainface = x.copy()

os.chdir('/Users/apple/Desktop/train/non-face')
filelist = os.listdir()
x = np.zeros((len(filelist),19*19))
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:] = np.array(IMG.getdata())
trainnonface = x.copy()


os.chdir('/Users/apple/Desktop/test/face')
filelist = os.listdir()
x = np.zeros((len(filelist),19*19))
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:] = np.array(IMG.getdata())
testface = x.copy()

os.chdir('/Users/apple/Desktop/test/non-face')
filelist = os.listdir()
x = np.zeros((len(filelist),19*19))
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:] = np.array(IMG.getdata())
testnonface = x.copy()


def BPNNtrain(pf,nf,hn,lr,iteration):
    pn = pf.shape[0]
    nn = nf.shape[0]
    fn = pf.shape[1]
    feature = np.append(pf,nf,axis=0)
    target = np.append(np.ones((pn,1)),np.zeros((nn,1)),axis=0)
    model = dict()
    WI = np.random.normal(0,1,(fn+1,hn))
    WO = np.random.normal(0,1,(hn+1,1))
    for t in range(iteration):
        s = random.sample(range(pn+nn),pn+nn)
        for i in range(pn+nn):
            ins = np.append(feature[s[i],:],1)
            ho = ins.dot(WI)
            for j in range(hn):
                ho[j] = 1/(1+math.exp(-ho[j]))
            hs = np.append(ho,1)
            out = hs.dot(WO)
            out = 1/(1+math.exp(-out))
            dk = out*(1-out)*(target[s[i]]-out)
            dh = ho*(1-ho)*WO[0:hn,0]*dk
            WO[:,0] = WO[:,0] + lr*dk*hs
            for j in range(hn):
                WI[:,j] = WI[:,j] + lr*dh[j]*ins
    model = dict()
    model['WI'] = WI
    model['WO'] = WO                                
    return model  

def BPNNtest(feature,model):
    sn = feature.shape[0]
    WI = model['WI']
    WO = model['WO']
    hn = WI.shape[1]
    out = np.zeros((sn,1))
    for i in range(sn):
        ins = np.append(feature[i,:],1)
        ho = ins.dot(WI)
        for j in range(hn):
            ho[j] = 1/(1+math.exp(-ho[j]))
        hs = np.append(ho,1)
        out[i] = hs.dot(WO)
        out[i] = 1/(1+math.exp(-out[i]))
    return out
"""

#改變hn
#network (4548 * 1)
#output (472 * 1)
#pscore (2429 * 1)
#nscore (4548 * 1)

"""
#第一張圖
#hn: 20 lr: 0.01 iteration: 10
network = BPNNtrain(trainface/255,trainnonface/255,20,0.01,10)
output = BPNNtest(testface/255,network)
pscore_train = BPNNtest(trainface/255,network)
nscore_train = BPNNtest(trainnonface/255,network)
pscore_test = BPNNtest(testface/255,network)
nscore_test = BPNNtest(testnonface/255,network)


#第二張圖
#hn: 30 lr: 0.01 iteration: 10


network = BPNNtrain(trainface/255,trainnonface/255,30,0.01,10)
output = BPNNtest(testface/255,network)
pscore_train =BPNNtest(trainface/255,network)
nscore_train = BPNNtest(trainnonface/255,network)
pscore_test = BPNNtest(testface/255,network)
nscore_test = BPNNtest(testnonface/255,network)


#第三張圖
#hn: 50 lr: 0.01 iteration: 10
            
network = BPNNtrain(trainface/255,trainnonface/255,50,0.01,10)
output = BPNNtest(testface/255,network)
pscore_train =BPNNtest(trainface/255,network)
nscore_train = BPNNtest(trainnonface/255,network)
pscore_test = BPNNtest(testface/255,network)
nscore_test = BPNNtest(testnonface/255,network)    

#改變lr

#第四張圖
#hn: 10 lr: 0.99 iteration: 10
network = BPNNtrain(trainface/255,trainnonface/255,20,0.99,10)
output = BPNNtest(testface/255,network)
pscore_train =BPNNtest(trainface/255,network)
nscore_train = BPNNtest(trainnonface/255,network)
pscore_test = BPNNtest(testface/255,network)
nscore_test = BPNNtest(testnonface/255,network)


#第五張圖
#hn: 10 lr: 0.5 iteration: 10
network = BPNNtrain(trainface/255,trainnonface/255,10,0.5,10)
output = BPNNtest(testface/255,network)
pscore_train =BPNNtest(trainface/255,network)
nscore_train = BPNNtest(trainnonface/255,network)
pscore_test = BPNNtest(testface/255,network)
nscore_test = BPNNtest(testnonface/255,network)
"""
#第六張圖
#hn: 10 lr: 0.25 iteration: 10
network = BPNNtrain(trainface/255,trainnonface/255,10,0.25,10)
output = BPNNtest(testface/255,network)    
pscore_train =BPNNtest(trainface/255,network)
nscore_train = BPNNtest(trainnonface/255,network)
pscore_test = BPNNtest(testface/255,network)
nscore_test = BPNNtest(testnonface/255,network) 
"""
            
#改變iteration 
#第七張圖
#hn: 10 lr: 0.01 iteration: 1000
network = BPNNtrain(trainface/255,trainnonface/255,10,0.01,500)
output = BPNNtest(testface/255,network)
pscore_train =BPNNtest(trainface/255,network)
nscore_train = BPNNtest(trainnonface/255,network)
pscore_test = BPNNtest(testface/255,network)
nscore_test = BPNNtest(testnonface/255,network)


#第八張圖
#hn: 10 lr: 0.01 iteration: 100
network = BPNNtrain(trainface/255,trainnonface/255,10,0.01,100)
output = BPNNtest(testface/255,network)
pscore_train =BPNNtest(trainface/255,network)
nscore_train = BPNNtest(trainnonface/255,network)
pscore_test = BPNNtest(testface/255,network)
nscore_test = BPNNtest(testnonface/255,network)


#第九張圖
#hn: 10 lr: 0.01 iteration: 30 
network = BPNNtrain(trainface/255,trainnonface/255,10,0.01,30)    
output = BPNNtest(testface/255,network)       
pscore_train =BPNNtest(trainface/255,network)
nscore_train = BPNNtest(trainnonface/255,network)
pscore_test = BPNNtest(testface/255,network)
nscore_test = BPNNtest(testnonface/255,network)
"""


parr_train = []
farr_train = []
parr_test = []
farr_test = []

TPR_train = []
FPR_train = []
TPR_test = []
FPR_test = []


for i in range(1,100,1):
    a = i/100
    parr_train.append(np.nonzero(pscore_train[:,0] > a)[0])
    farr_train.append(np.nonzero(nscore_train[:,0] < a)[0])
    
    parr_test.append(np.nonzero(pscore_test[:,0] > a)[0])
    farr_test.append(np.nonzero(nscore_test[:,0] < a)[0])
    
    TPR_train.append(len(parr_train[i-1]) / len(pscore_train))
    FPR_train.append((len(nscore_train) - len(farr_train[i-1])) / len(nscore_train))
    
    TPR_test.append(len(parr_test[i-1]) / len(pscore_test))
    FPR_test.append((len(nscore_test) - len(farr_test[i-1])) / len(nscore_test))

plt.plot(FPR_train,TPR_train)
plt.plot(FPR_test,TPR_test)

plt.show()

  
    
    
    
    