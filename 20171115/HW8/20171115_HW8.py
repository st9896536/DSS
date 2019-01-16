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



"""
def F1(t): #Signal generator 線性迴歸
    return 0.063*(t**3) - 5.284 *t *t + 4.887 * t + 412 + np.random.normal(0,1)

def F2(t,A,B,C,D): #Signal generator 非線性迴歸
    return A * (t**B) + C*np.cos(D*t) + np.random.normal(0,1,t.shape)

def Energy(b2,A2,A,B,C,D): #找出最好的ＡＢＣＤ
    return np.sum(abs(F2(A2,A,B,C,D)-b2))
"""


def F2(A_new,A,B,C,tc,beta,omega,phi): #Signal generator 非線性迴歸
    num = []
    for t in range(tc):
        num.append(A+B*((tc-A_new[t])**beta)+C*(((tc-A_new[t])**beta)*(math.cos(omega*(math.log2(tc-A_new[t]))+phi))))
    num = np.array(num)
    return num


def Energy(A_new,b_new,tc,beta,omega,phi,A,B,C): #找出最好的七個參數
    predict = F2(A_new,A,B,C,tc,beta,omega,phi)
    return np.sum(abs(predict-b_new)) #abs表絕對值


def returnABC(tc,beta,omega,phi):
    b_new = np.zeros((tc,1)) #Ax = b 已知A 和 b 求 x 
    A_new = np.zeros((tc,3)) 
    for i in range(tc):
        b_new[i] = data_price[i]
        A_new[i,0] = (tc-i) ** beta * (math.cos(omega * math.log(tc-i) + phi)) #C
        A_new[i,1] = (tc-i) ** beta #B
        A_new[i,2] = 1 #A
    X = np.linalg.lstsq(A_new,b_new)[0] #線性迴歸
    C = X[0][0]
    B = X[1][0]
    A = X[2][0]
    return A, B, C

def Gettc(tc):
    
    return (643*tc + 643 -550*tc + 550*1024)//(1025)


#------------------------------------------
"""
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
        exp2[i][j] = Energy(b2,A2,i/100,1.2,j/100,0.4)

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


n = 644
data_price = np.zeros((n,1))
date = np.zeros((n,1))
for i in range(n): #data 的 length
    data_price[i,:] = data[i,1] #把data的股價load進來
    date[i,:] = data[i,0] #把data的時間load進來
    

pop = np.random.randint(0,2,(1000,40)) #1000*40的array
#適者生存
fit = np.zeros((1000,1)) #1000個人
for generation in range(10): #砍到剩10個人
    print(generation)
    #留下活得好的人 留下好的人繁衍 把990個人生回來 
    for i in range(1000):
        gene = pop[i,:]
        #產生2的0次方 ～ 2的9次方
        tc = 550 + (np.sum(2**np.array(range(6))*gene[0:6]) - 32)
        tc = Gettc(tc)
        beta = (np.sum(2**np.array(range(10))*gene[6:16])) / 1024
        omega = (np.sum(2**np.array(range(14))*gene[16:30])) * 0.03
        phi = ((np.sum(2**np.array(range(10))*gene[30:40])) / 1024) * 2 * math.pi
        
        A,B,C = returnABC(tc,beta,omega,phi)
        
        t2 = np.zeros((tc,1))
        output = np.zeros((tc,1))
        
        for t in range(tc):
            t2[t] = t
            output[t] = data_price[t]
        
        fit[i]=Energy(t2,output,tc,beta,omega,phi,A,B,C)
        
    sortf = np.argsort(fit[:,0]) #把10000個人裡面排序 fit越小的會排得越前面
    pop = pop[sortf,:] #前100個不要動 第101個從前100個人裡面生
    for i in range(100,1000):
        fid = np.random.randint(0,100)
        mid = np.random.randint(0,100)
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
    tc = 550 + (np.sum(2**np.array(range(6))*gene[0:6]) - 32)
    tc = Gettc(tc)
    beta = (np.sum(2**np.array(range(10))*gene[6:16])) / 1024
    omega = (np.sum(2**np.array(range(14))*gene[16:30])) * 0.03
    phi = ((np.sum(2**np.array(range(10))*gene[30:40])) / 1024) *2 * math.pi
        
    A,B,C = returnABC(tc,beta,omega,phi)
    
    t2 = np.zeros((tc,1))
    result = np.zeros((tc,1))
        
    for t in range(tc):
        t2[t] = t
        result[t] = data_price[t]
        
    fit[i]=Energy(t2,result,tc,beta,omega,phi,A,B,C)
        
    
sortf = np.argsort(fit[:,0]) #把10000個人裡面排序 fit越小的會排得越前面
pop = pop[sortf,:] #前100個不要動 第101個從前100個人裡面生

gene = pop[0,:]
tc = 550 + (np.sum(2**np.array(range(6))*gene[0:6]) - 32)
tc = Gettc(tc)
beta = (np.sum(2**np.array(range(10))*gene[6:16])) / 1024
omega = (np.sum(2**np.array(range(14))*gene[16:30])) * 0.03
phi = ((np.sum(2**np.array(range(10))*gene[30:40])) / 1024) *2 * math.pi

A,B,C = returnABC(tc,beta,omega,phi)

actual = []
time = []
answer = []

for t in range(tc):
    actual.append(data_price[t]) #長到tc為止的price資料
    time.append(t) #長到tc為止的時間

actual = np.array(actual)

error = Energy(time,actual,tc,beta,omega,phi,A,B,C)

answer.append(F2(time,A,B,C,tc,beta,omega,phi))
answer = np.array(answer)
answer = answer.T



print("tc = %d" % tc )
print("beta = %2f "  % beta )
print("omega = %2f" % omega )
print("phi = %2f" % phi )
print("A = %2f" % A )
print("B = %2f" % B )
print("C = %2f" % C )
print("Avg_error = %2f" % (error / tc))


plt.plot(data[:,1],'b') #把Bubble的圖load進來
plt.plot(time, answer,'g')
plt.show()

         

