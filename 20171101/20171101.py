#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 19:15:43 2017

@author: Rong
"""

#把19 * 19的照片拉成一維 共有2429張照片 把他load進資料庫
import os
from PIL import Image
import numpy as np
import random
import math
import matplotlib.pyplot as plt

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


#hn: 幾個hidden layer lr: learning rate只能自己設定（更新權重，學新的東西的百分比）
def BPNN(pf,nf,hn,lr,iteration):
    pn = pf.shape[0]  #正資料  2429*361
    nn = nf.shape[0]  #負資料 4548*361
    fn = pf.shape[1]  #feature number
    
    feature = np.append(pf,nf,axis=0)
    target = np.append(np.ones((pn,1)),np.zeros((nn,1)),axis = 0) #先把標準答案的矩陣建起來
    #建構網路
    WI = np.random.normal(0,1,(fn+1,hn)) #第一層權重 WI:weight input node
    WO = np.random.normal(0,1,(hn+1,1)) #WO:
    
    for t in range(iteration):
        s = random.sample(range(pn+nn),pn+nn)
        for i in range(pn+nn):
            ins = np.append(feature[s[i],:],1) #特徵值拿出來給 input signal 在冠上1 361個特徵
            ho = ins.dot(WI) #做矩陣相乘
            for j in range(hn):
                ho[j] = 1/(1+math.exp(-ho[j])) 
            hs = np.append(ho,1) #hidden node signal 第二層 20個值 冠 1
            out = hs.dot(WO)
            dk = out * (1 - out) *(target[s[i]] - out) #可以得到最後一個node 的 delta值
            dh = np.zeros((hn,1)) #hidden node 的梯度？
            for j in range(hn): #幫每個hidden node 跟 output node 算出delta 值
                dh[j] = ho[j] * (1 - ho[j]) * WO[j] * dk
            WO = WO + lr * dk * hs
            for j in range(hn):
                WI[:,j] = WI[:,j] + lr * dh[j] * ins #整排一起更新 362*20 20 * 1
    model = dict()
    model['WI'] = WI
    model['WO'] = WO
    return model         

def BPNNtest(feature,model):
    sn = feature.shape[0]
    WI = model['WI']
    WO = model['WO']
    for i in range(sn):
        ins = np.append(feature[i,:],1)
        ho = ins.dot(WI)
        for j in range(sn):
            ho[j] = 1/(1+math.exp(-ho[j])) 
        hs = np.append(ho,1) #hidden node signal 第二層 20個值 冠 1
        out = hs.dot(WO)
    return out


#改變hn


network1 = BPNN(trainface/255,trainnonface/255,30,0.2,10)
output1 = BPNNtest(testface/255,network1)

network2 = BPNN(trainface/255,trainnonface/255,20,0.2,10)
output2 = BPNNtest(testface/255,network2)
            
network3 = BPNN(trainface/255,trainnonface/255,10,0.2,10)
output3 = BPNNtest(testface/255,network3)     

#改變lr
network4 = BPNN(trainface/255,trainnonface/255,10,0.5,10)
output4 = BPNNtest(testface/255,network4)

network5 = BPNN(trainface/255,trainnonface/255,10,0.8,10)
output5 = BPNNtest(testface/255,network5)

network6 = BPNN(trainface/255,trainnonface/255,10,0.2,10)
output6 = BPNNtest(testface/255,network6)     
            
#改變iteration 

network7 = BPNN(trainface/255,trainnonface/255,10,0.2,10)
output7 = BPNNtest(testface/255,network7)

network8 = BPNN(trainface/255,trainnonface/255,10,0.2,100)
output8 = BPNNtest(testface/255,network8)

network9 = BPNN(trainface/255,trainnonface/255,10,0.2,50)    
output9 = BPNNtest(testface/255,network9)       
