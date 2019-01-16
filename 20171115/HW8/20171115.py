#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:32:57 2017

@author: Rong
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
data = np.loadtxt('/Users/apple/Desktop/Bubble.txt',delimiter=' ')

plt.plot(data[:,1]) #把Bubble的圖load進來
plt.show()

def F1(t): #Signal generator 線性迴歸
    return 0.063*(t**3) - 5.284 *t *t + 4.887 * t + 412 + np.random.normal(0,1)

def F2(t,A,B,C,D): #Signal generator 非線性迴歸
    return A * (t**B) + C*np.cos(D*t) + np.random.normal(0,1,t.shape)

def Energy(b2,A2,A,B,C,D): #找出最好的ＡＢＣＤ
    return np.sum(abs(F2(A2,A,B,C,D)-b2))
    


"""
n = 1000 #sample數量越多誤差會越小 會越接近signal
b = np.zeros((n,1)) #Ax = b 已知A 和 b 求 x
A3 = np.zeros((n,5))
for i in range(n):
    t = np.random.random() * 100
    b[i] = F1(t)  #取出A和b矩陣
    A3[i,0] = t ** 4
    A3[i,1] = t ** 3
    A3[i,2] = t ** 2
    A3[i,3] = t
    A3[i,4] = 1
    
x = np.linalg.lstsq(A3,b)[0] #線性迴歸
print(x)

#------------------------------------------
n = 1000
b2 = np.zeros((n,1))
A2 = np.random.random((n,1))*100
b2 = F2(A2,0.6,1.2,100,0.4)

print(Energy(b2,A2,0.6,1.2,100,0.4))  #1155
print(Energy(b2,A2,0.6,1.2,99,0.5)) #80873


#----------------第一題---------------------
exp1 = []
#n = 1000
#b2 = np.zeros((n,1))
#A2 = np.zeros((n,1))
for i in range(-511,513,1):
    i = i / 100
    #print(Energy(b2,A2,0.6,1.2,100,i))
    exp1.append(Energy(b2,A2,0.6,1.2,100,i))  
    
#exp_x = list(range(float(-512/100),float(513/100)))
exp1_x = np.linspace(-5.12,5.13,1024)

plt.plot(exp1_x,exp1)
plt.show()
    

#----------------第二題---------------------

#固定B, D
exp2 = np.zeros((1024,1024))

for i in range(-511,513,1):
    for j in range(-511,513,1):
        exp2[][j] = Energy(b2,A2,i/100,1.2,j/100,0.4)

exp2_A = np.linspace(-5.12,5.13,1024)
exp2_C = np.linspace(-512,513,1024)

for i in range(len(exp2_A)):
    for j in range(len(exp2_C)):
        exp2[i][j] = Energy(b2,A2,exp2_A[i],1.2,exp2_C[j],0.4)

#Axes3D.plot_surface(exp2_A, exp2_C, exp2)

A, C = np.meshgrid(exp2_A, exp2_C)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, C, exp2)
plt.show()
"""

#---------------基因演算法------------------

"""
pop = np.random.randint(0,2,(1000,40))
#適者生存
fit = np.zeros((1000,1)) #10000個人
for generation in range(10): #砍到剩100個人
    print(generation)
    #留下活得好的人 留下好的人繁衍 把9900個人生回來 
    for i in range(1000):
        gene = pop[i,:]
        #產生2的0次方 ～ 2的9次方
        gene = pop[i,:]
        A = (np.sum(2**np.array(range(10))*gene[0:10])-511)/100
        B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
        C = np.sum(2**np.array(range(10))*gene[20:30])-511
        D = (np.sum(2**np.array(range(10))*gene[30:40])-511)/100
        fit[i]=Energy(b2,A2,A,B,C,D)
    sortf = np.argsort(fit[:,0]) #把10000個人裡面排序 fit越小的會排得越前面
    pop = pop[sortf,:] #前100個不要動 第101個從前100個人裡面生
    for i in range(100,1000):
        fid = np.random.random(0,100)
        mid = np.random.random(0,100)
        while(mid == fid):
            mid = np.random.randint(0,100)
        mask = np.random.randint(0,2,(1,40))
        son = pop[mid,:]
        father = pop[fid,:]
        son[mask[0,:]==1]=father[mask[0,:]==1]
        pop[i,:] = son
    for i in range(100): #基因突變
        m = np.random.randint(0,1000)
        n = np.random.randint(0,40)
        if(pop[m,n]==0):
            pop[m,n] = 1
        else:
            pop[m,n] = 0
    

    

for generation in range(100):
    gene = pop[i,:]
    #產生2的0次方 ～ 2的9次方
    A = (np.sum(2 ** np.array(range(10))*gene[0:10]) - 511) / 100
    B = (np.sum(2 ** np.array(range(10))*gene[10:20]) - 511) / 100
    C = (np.sum(2 ** np.array(range(10))*gene[20:30]) - 511)
    D = (np.sum(2 ** np.array(range(10))*gene[30:40]) - 511) / 100
    fit[i] = Energy(b2,A2,A,B,C,D)
sortf = np.argsort(fit[:0]) #把10000個人裡面排序 fit越小的會排得越前面
pop = pop[sortf,:] #前100個不要動 第101個從前100個人裡面生

gene = pop[0,:]
A = (np.sum(2**np.array(range(10))*gene[0:10])-511)/100
B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
C = np.sum(2**np.array(range(10))*gene[20:30])-511
D = (np.sum(2**np.array(range(10))*gene[30:40])-511)/100

    
"""
