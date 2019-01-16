#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:51:42 2017

@author: Rong
"""

import numpy as np
import matplotlib.pyplot as plt
#import math
import random
from sklearn import datasets
from sklearn.metrics import confusion_matrix
iris = datasets.load_iris()


#D:dimension
def kmeans(sample,K,maxiter): # N 是sample的數量
    N = sample.shape[0] # 矩陣的長為N
    D = sample.shape[1] # 矩陣的寬為D
    C = np.zeros((K,D)) #一個K*D的矩陣 裡面都填0 
    L = np.zeros((N,1)) #更新前 L(label)有點像target 如果都是1的話就放在G1裡面
    L1 = np.zeros((N,1)) #更新後
    dist = np.zeros((N,K))
    idx = random.sample(range(N),K) #從N個點裡面取出K個中心點
    C = sample[idx,:] #一開始預設的中心點 center
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

target = iris.target
data = iris.data

G = data  #把鳶尾花的data傳給G1、G2
G_scale = data

#算normalization standard score  #data - mean / std
G = (G - np.tile(G.mean(0),(G.shape[0],1))) / np.tile(G.std(0),(G.shape[0],1))

#算scaling normalization # data - min / max - min

for i in range(G_scale.shape[1]):
    minv = np.min(G_scale[:,i])
    maxv = np.max(G_scale[:,i])
    G_scale[:,i] = (G_scale[:,i]-minv)/ (maxv - minv)



C,L,wicd1 = kmeans(G,3,5000)
#先假設這次的wicd是最好的
minwicd = wicd1
bestC = C
bestL = L

for i in range(50):
    C,L,wicd1 = kmeans(G,3,5000)
    if wicd1 < minwicd:
        minwicd = wicd1 #把最好的wicd傳給minwicd
        bestC = C
        bestL = L


C,L,wicd2 = kmeans(G_scale,3,5000)
minwicd_scale = wicd2
bestC_scale = C
bestL_scale = L
for i in range(50):
    C,L,wicd2 = kmeans(G_scale,3,5000)
    if wicd2 < minwicd_scale:
        minwicd_scale = wicd2 
        bestC_scale = C
        bestL_scale = L
        

#----------------------------------------------------
#-----------------------第二題------------------------
#算出每個點到點的distance
dists = np.zeros((150,150))
distidx = np.zeros((150,150))

for i in range(150):
    for j in range(150):
        dist = np.sum(np.sqrt(np.square(G[i] - G[j]))) #np.square = ** 2
        dists[i,j] = dist
        #對已經算好的distance做排序
        distidx = np.argsort(dists)
        distidx = np.delete(distidx, 0, 1)


#投票區 
#建一個3 * 3 array裡面都放0
metrics = [[0 for x in range(3)] for y in range(3)] 

predict = [0] * 150
for i in range(150):
    for k in range(10): #1NN ~ KNN
        vote = target[distidx[:,:k+1]] #將前10筆投票結果給vote
    count = np.bincount(vote[i,:])
    
    predict[i] = np.argmax(count) #將150筆投票結果傳給predict
    
metrics = confusion_matrix(predict, target)
print(metrics)


"""
    # 檢查predict 和 label 建立metrics
    #---------如果都預測對的話
    if target[i] and predict[i] == 0:
        metrics[0][0] = metrics[0][0] + 1
    elif target[i] and predict[i] == 1:
        metrics[1][1] = metrics[1][1] + 1
    elif target[i] and predict[i] == 2:
        metrics[2][2] = metrics[2][2] + 1
    #---------如果label是0但predict錯的話
    elif target[i] == 0 and predict[i] == 1:
        metrics[0][1] = metrics[0][1] + 1
    elif target[i] == 0 and predict[i] == 2:
        metrics[0][2] = metrics[0][2] + 1
    #---------如果label是1但predict錯的話
    elif target[i] == 1 and predict[i] == 0:
        metrics[1][0] = metrics[1][0] + 1
    elif target[i] == 1 and predict[i] == 2:
        metrics[1][2] = metrics[1][2] + 1
    #---------如果label是2但predict錯的話
    elif target[i] == 2 and predict[i] == 0:
        metrics[2][0] = metrics[2][0] + 1
    elif target[i] == 2 and predict[i] == 1:
        metrics[2][1] = metrics[2][1] + 1

metricsarray = np.asarray(metrics)    

"""

#----------------------------------------------------




print("ormalization standard score wicd: ",minwicd)
print("scaling normalization wicd: ",minwicd_scale)

"""
G1 = G[bestL==0,:]
G2 = G[bestL==1,:]
G3 = G[bestL==2,:]
"""

G1 = G_scale[bestL==0,:]
G2 = G_scale[bestL==1,:]
G3 = G_scale[bestL==2,:]


plt.plot(G1[:,0],G1[:,1],'r.',G2[:,0],G2[:,1],'g.',G3[:,0],G3[:,1],'b.')
plt.plot(C[:,0],C[:,1],'kx',)




