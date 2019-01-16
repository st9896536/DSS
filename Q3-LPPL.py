# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:49:59 2017

@author: hwang
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

#data = np.loadtxt('Bubble.txt') 
data= pd.read_csv('Bubble.txt',sep=' ',header=None)

def F2(A3,A,B,C,tc,beta,omega,phi):
    ans = []
    for i in range(len(A3)):
        ans.append(A+B*((tc-A3[i])**beta)+C*(((tc-A3[i])**beta)*(math.cos(omega*(math.log2(tc-A3[i]))+phi))))
    ans = np.array(ans)
    return ans

def LPPL(b2,A3,A,B,C,tc,beta,omega,phi):
    ans=F2(A3,A,B,C,tc,beta,omega,phi)
    return np.sum(abs(ans-b2))

def getTc(Tc):

    return (643*Tc + 643 -550*Tc + 550*1024)//(1025)
    

def computeABC(Tc,Beta,Phi,Omega):
    A3 = np.zeros((Tc,3))
    b = np.zeros((Tc,1))
    for t in range(Tc):
        b[t] = b2[t]
        A3[t,0] = 1
        A3[t,1] = (Tc-t)**Beta
        A3[t,2] = ((Tc-t)**Beta)*math.cos(Omega*(math.log2(Tc-t))+Phi)
    x = np.linalg.lstsq(A3,b)[0]
    A = x[0][0]
    B = x[1][0]
    C = x[2][0]

    return A,B,C
    
n = 1000
b2 = data[1]
t = data[0]

date = []

for i in range(t.shape[0]):
    date.append(i)
    
    
date = np.array(date)

   
pop = np.random.randint(0,2,(1000,40))
fit = np.zeros((1000,1))


for generation in range(10):
    print(generation)
    
    for i in range(1000):
        gene = pop[i,:]
        Tc =(np.sum(2**np.array(range(10))*gene[0:10]))
        Tc =getTc(Tc)
        Omega = (np.sum(2**np.array(range(10))*gene[10:20]))*10/1024
        Beta = np.sum(2**np.array(range(10))*gene[20:30])/1024
        Phi = (np.sum(2**np.array(range(10))*gene[30:40]))/1024*6.28
        
        A,B,C = computeABC(Tc,Beta,Phi,Omega)
        
        t2 = np.zeros((Tc,1))
        output = np.zeros((Tc,1))
        
        for k in range(Tc):
            t2[k]=k
            output[k]=b2[k]
            
        fit[i]=LPPL(output,t2,A,B,C,Tc,Beta,Omega,Phi)
        
    sortf = np.argsort(fit[:,0])
    pop = pop[sortf,:]
    for i in range(100,1000):
        fid = np.random.randint(0,100)
        mid = np.random.randint(0,100)
        while(mid==fid):
            mid = np.random.randint(0,100)
        mask = np.random.randint(0,2,(1,40))
        son = pop[mid,:]
        father = pop[fid,:]
        son[mask[0,:]==1]=father[mask[0,:]==1]
        pop[i,:] = son
    for i in range(100):
        m = np.random.randint(0,1000)
        n = np.random.randint(0,40)
        if(pop[m,n]==0):
            pop[m,n]=1
        else:
            pop[m,n]=0


for i in range(1000):
    
        gene = pop[i,:]
        Tc =(np.sum(2**np.array(range(10))*gene[0:10]))
        Tc = getTc(Tc)
        Omega = (np.sum(2**np.array(range(10))*gene[10:20]))*10/1024
        Beta = np.sum(2**np.array(range(10))*gene[20:30])/1024
        Phi = (np.sum(2**np.array(range(10))*gene[30:40]))/1024*6.28
        
        A,B,C = computeABC(Tc,Beta,Phi,Omega)
        
        t2 = np.zeros((Tc,1))
        predict = np.zeros((Tc,1))
        
        for k in range(Tc):
            t2[k]=k
            predict[k]=b2[k]
            
        fit[i]=LPPL(predict,t2,A,B,C,Tc,Beta,Omega,Phi)
        
sortf = np.argsort(fit[:,0])
pop = pop[sortf,:]

gene = pop[0,:]
Tc =(np.sum(2**np.array(range(10))*gene[0:10]))
Tc = getTc(Tc)
Omega = (np.sum(2**np.array(range(10))*gene[10:20]))*10/1024
Beta = np.sum(2**np.array(range(10))*gene[20:30])/1024
Phi = (np.sum(2**np.array(range(10))*gene[30:40]))/1024*6.28
A,B,C = computeABC(Tc,Beta,Phi,Omega)


actualdata = []
resultarr = []
resultT = []
for i in range(Tc):
    actualdata.append(b2[i])
    resultT.append(i)
actualdata = np.array(actualdata)

error = LPPL(actualdata,resultT,A,B,C,Tc,Beta,Omega,Phi)


resultarr.append(F2(resultT,A,B,C,Tc,Beta,Omega,Phi))
resultarr = np.array(resultarr)
resultarr = resultarr.T  #轉置矩陣



print("A: %.2f" % A)
print("B: %.2f" % B)
print("C: %.2f" % C)
print("Beta: %.2f" %Beta)
print("Omega: %.2f" %Omega)
print("Phi: %.2f" %Phi)
print("Tc: %.2f" %Tc)
print("Avgerror: %.2f" % (error/Tc))



plt.plot(date,data[1],'b')
plt.plot(resultT,resultarr,'r')
plt.show()
 

