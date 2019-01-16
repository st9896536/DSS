#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:02:08 2017

@author: Rong
"""

import math
from scipy.stats import norm
import matplotlib.pyplot as plt
#import numpy as np

def blscall(S,L,T,r,sigma):
    d1 = (math.log(S/L)+(r + 0.5 * sigma * sigma) * T)/(sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - L * math.exp(-r * T) * norm.cdf(d2)

def Bisection(left,right,S,L,T,r,call,error,iteration): #用二分法求逼近解
    center = (left+right)/2
    Bigraph.append(center)
    if iteration == 0:
        return center
    if((blscall(S,L,T,r,left)-call)* (blscall(S,L,T,r,center)-call)<0):
        return Bisection(left,center,S,L,T,r,call,error,iteration-1)
    else:
        return Bisection(center,right,S,L,T,r,call,error,iteration-1)
    
    
def Newton(initsigma,S,L,T,r,call,iteration):
    sigma = initsigma
    for i in range(iteration):
        Newgraph.append(sigma)
        fx = blscall(S,L,T,r,sigma) - call
        fx2 = (blscall(S,L,T,r,sigma+0.00000001) - blscall(S,L,T,r,sigma-0.00000001))/0.00000002
        sigma = sigma - fx/fx2
    return sigma
        
    
    
# 第三題
Bigraph = []
Newgraph = []
listt = []

S = 10326.68
L = 10300.0 
T = 21.0/365 #到期日10/19 （今日：9/27）
r = 0.01065 #台灣銀行定存利率
sigma = 0.10508 #波動率 數字越大選擇權價值越高 # 唯一大家不知道的參數

#print(blscall(S,L,T,r,sigma))
Bigraph.append(Bisection(0.0000001,1,S,L,T,r,121.0,0.0000001,20))
Newgraph.append(Newton(0.5,S,L,T,r,121.0,20))
    
#print(blscall(S,L,T,r,sigma))
#print(Newton(0.5,S,L,T,r,121.0,20))


plt.plot(Bigraph)
plt.plot(Newgraph)
plt.show()

#第四題


L = 10000.0 

listt.append(Newton(0.5,S,L,T,r,354.0,20))

L = 10100.0
listt.append(Newton(0.5,S,L,T,r,267.0,20))

L = 10200.0
listt.append(Newton(0.5,S,L,T,r,187.0,20))

L = 10300.0
listt.append(Newton(0.5,S,L,T,r,121.0,20))

L = 10400.0
listt.append(Newton(0.5,S,L,T,r,69.0,20))

L = 10500.0
listt.append(Newton(0.5,S,L,T,r,34.0,20))

L = 10600.0
listt.append(Newton(0.5,S,L,T,r,14.50,20))
L = 10700.0
listt.append(Newton(0.5,S,L,T,r,5.90,20))

plt.plot(listt)
plt.show()