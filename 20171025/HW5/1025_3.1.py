#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:01:50 2017

@author: Rong
"""

#----------------------------------------------
#策略3.1
#找到一組m, n改進策略2.1使總損益點數最佳化
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('TXF20112015.csv',sep=',',header=None)
TAIEX = df.values  #數字陣列
tradeday = list(set(TAIEX[:,0]//10000)) #把日期欄位拿出來除以10000 8碼的日期 年年月月日日
tradeday.sort()

maxprofit2 = 0
maxm = 0
maxn = 0
maxprofit = []

profit = np.zeros((len(tradeday),1))

for m in range(10,110,10): #停利
    for n in range(10,m+10,10): #停損
        for i in range(len(tradeday)):
            date = tradeday[i]
            idx = np.nonzero(TAIEX[:,0]//10000==date)[0] 
            idx.sort()
            p1 = TAIEX[idx[0],2] #一開盤的第一分鐘開盤價
            idx2 = np.nonzero(TAIEX[idx,3] >= p1 + n)[0] #停損
            idx3 = np.nonzero(TAIEX[idx,4] <= p1 - m)[0] #停利
            if len(idx2)==0 and len(idx3)==0: #如果沒有停利根停損
                p2 = TAIEX[idx[-1],1] #如果沒有的話 以當天的收盤價收盤
            elif len(idx3) == 0: 
                p2 = TAIEX[idx[idx2[0]],1]  
            elif len(idx2) == 0:
                p2 = TAIEX[idx[idx3[0]],1]
            elif idx2[0]<idx3[0]:  #停損比停利早出現
                p2 = TAIEX[idx[idx2[0]],1]
            else:
                p2 = TAIEX[idx[idx3[0]],1]
            profit[i] = -(p2-p1)
        profit2 = np.cumsum(profit) #累加第一天到最後一天的profit 放到profit2裡面
        if profit2[-1] > maxprofit2:
            maxprofit2 = profit2[-1] #先讓目前的profit2的最後一筆存進maxprofit2
            maxm = m
            maxn = n
            maxprofit = profit
            
for i in range(len(tradeday)):
            date = tradeday[i]
            idx = np.nonzero(TAIEX[:,0]//10000==date)[0] 
            idx.sort()
            p1 = TAIEX[idx[0],2] #一開盤的第一分鐘開盤價
            idx2 = np.nonzero(TAIEX[idx,3] >= p1 + 20)[0] #停損
            idx3 = np.nonzero(TAIEX[idx,4] <= p1 - 100)[0] #停利
            if len(idx2)==0 and len(idx3)==0: #如果沒有停利根停損
                p2 = TAIEX[idx[-1],1] #如果沒有的話 以當天的收盤價收盤
            elif len(idx3) == 0: 
                p2 = TAIEX[idx[idx2[0]],1]  
            elif len(idx2) == 0:
                p2 = TAIEX[idx[idx3[0]],1]
            elif idx2[0]<idx3[0]:  #停損比停利早出現
                p2 = TAIEX[idx[idx2[0]],1]
            else:
                p2 = TAIEX[idx[idx3[0]],1]
            profit[i] = -(p2-p1)
profit2 = np.cumsum(profit) #累加第一天到最後一天的profit 放到profit2裡面
        
                
plt.plot(profit2)
plt.show()

ans1 = profit2 #總損益點數士profit2的最後一筆數值
ans2 = np.sum(profit>0)/len(profit) #勝率 profit>0的個數除以總個數
ans3 = np.mean(profit[profit>0]) #賺錢時平均每次獲利點數
ans4 = np.mean(profit[profit<=0]) #輸錢時平均每次獲利點數
plt.hist(profit,bins=100) #繪出每日損益的分布圖
plt.show()

print('3.1: ',ans1,ans2,ans3,ans4)
print('m: ',maxm)
print('n: ',maxn)
#3.1:  1859.0 0.494308943089 45.4490131579 -42.9469453376
#m:  100
#n:  20
