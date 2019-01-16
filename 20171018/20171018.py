#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:51:42 2017

@author: Rong
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn import datasets
iris = datasets.load_iris()


target = iris.target

#D:dimension
def kmeans(sample,K,maxiter): #N 是sample的數量
    N = sample.shape[0]
    D = sample.shape[1]
    C = np.zeros((K,D)) #K*D
    L = np.zeros((N,1)) #更新前 L有點像target 如果都是1的話就放在G1裡面
    L1 = np.zeros((N,1)) #更新後
    dist = np.zeros((N,K))
    idx = random.sample(range(N),K) #從N個點裡面取出K個中心點
    C = sample[idx,:] #一開始預設的中心點
    iter = 0
    while(iter<=maxiter): #檢查iter有沒有超過
        for i in range(K): #每一個center拿出來
            dist[:,i] = np.sum((sample - np.tile(C[i,:],(N,1)))**2,1)
        L1 = np.argmin(dist,axis=1) #告訴你橫向裡面哪個是最小的
        if iter>0 and np.array_equal(L,L1):
            break;  #如果這一次跟上一次的label一樣的話 就中斷
        L = L1
        for i in range(K):
            idx = np.nonzero(L==i)[0] #回傳index
            if(len(idx)>0): #如果有人投票給我的話
                C[i,:] = np.mean(sample[idx,:],0) #把那些有投票給我取mean 0 往下算 之後會用到新的center
        iter += 1
    
    wicd = np.sum(np.sqrt(np.sum((sample - C[L,:])**2,1)))   
    return C,L,wicd #最後一趟的center和label 
        #每一個sample減center的 distance最好的留起來



#mean [0,0] 標準差[1,1]

G1 = np.random.normal(0,1,(5000,2))  #產生一個5000*2的矩陣 裡面放亂數
G1 = G1 + 4

G2 = np.random.normal(0,1,(3000,2)) #＊3 標準差會乘以3倍 點會被拉長
G2[:,1] = G2[:,1] * 3 - 3

G = np.append(G1,G2,axis = 0)

#G3 是y軸先拉長
G3 = np.random.normal(0,1,(2000,2))
G3[:,1] = G3[:,1] * 4

#然後在逆時鐘旋轉45度
c45 = math.cos(-45/180*math.pi)
s45 = math.sin(-45/180*math.pi)
R = np.array([[c45,-s45],[s45,c45]])

#再shift到[-4,6]的位置上
G3 = G3.dot(R)
G3[:,0] = G3[:,0] - 4 # x 移到 -4
G3[:,1] = G3[:,1] + 6 # y 移到 6

G = np.append(G,G3,axis = 0)
plt.plot(G[:,0],G[:,1],'.')


C,L,wicd = kmeans(G,3,1000)
G1 = G[L==0,:]
G2 = G[L==1,:]
G3 = G[L==2,:]


plt.plot(G1[:,0],G1[:,1],'r.',G2[:,0],G2[:,1],'g.',G3[:,0],G3[:,1],'b.')
plt.plot(C[:,0],C[:,1],'kx',)

print(wicd)

G = (G - np.tile(G.mean(0),(G.shape[0],1))) / np.tile(G.std(0),(G.shape[0],1))


for i in range(G.shape[1]):
    meanv = np.mean(G[:,i])
    stdv = np.std(G[:,i])
    G[:,i] = (G[:,i]-meanv)/stdv
    
    
    
    
