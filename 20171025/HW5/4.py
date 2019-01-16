#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:12:54 2017

@author: Rong
"""
# Time 收 開 高 低
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('TXF20112015.csv',sep=',',header=None)
TAIEX = df.values  #數字陣列
tradeday = list(set(TAIEX[:,0]//10000)) #把日期欄位拿出來除以10000 8碼的日期 年年月月日日
tradeday.sort()


#----------------------------------------------
#策略4
#開盤後，若先到達分鐘最高價高於開盤+30，則買進一口，抱到收盤平倉，若先到達分鐘最低價低於開盤-30，則放空一口，一樣收盤平倉
count = 0 #進場次數
profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0] 
    idx.sort()
    
    p1 = TAIEX[idx[0],2] #一開盤的第一分鐘開盤價
    
    idx2 = np.nonzero(TAIEX[idx,4] <= p1-30)[0] #最低價低於開盤-30點 放空一口
    idx3 = np.nonzero(TAIEX[idx,3] >= p1+30)[0] #最高價高於開盤+30點 買一口 
    
    if len(idx2)==0 and len(idx3)==0: #如果沒有停利根停損
        profit[i] = 0
    elif len(idx2) == 0: #如果有買一口
        p1 = TAIEX[idx[idx3[0]],1]
        p2 = TAIEX[idx[-1],1]  #那一天的最後收盤價格
        profit[i] = p2 - p1
        count += 1
    elif len(idx3) == 0:  #如果有放空一口
        p1 = TAIEX[idx[idx2[0]],1]
        p2 = TAIEX[idx[-1],1]  #那一天的最後收盤價格
        profit[i] = -(p2 - p1)
        count += 1
    elif idx3[0]<idx2[0]:
        p1 = TAIEX[idx[idx3[0]],1]
        p2 = TAIEX[idx[-1],1]  #那一天的最後收盤價格
        profit[i] = p2 - p1
        count += 1
    else:
        p1 = TAIEX[idx[idx2[0]],1]
        p2 = TAIEX[idx[-1],1]  #那一天的最後收盤價格
        profit[i] = -(p2 - p1)
        count += 1
        

profit2 = np.cumsum(profit) #累加第一天到最後一天的profit 放到profit2裡面
plt.plot(profit2)
plt.show()

ans1 = profit2[-1] #總損益點數士profit2的最後一筆數值
ans2 = np.sum(profit>0)/count #勝率 profit>0的個數除以總個數
ans3 = np.mean(profit[profit>0]) #賺錢時平均每次獲利點數
ans4 = np.sum(profit[profit<0])/(count-np.sum(profit>0)) #輸錢時平均每次獲利點數
plt.hist(profit,bins=100) #繪出每日損益的分布圖
plt.show()

print('4: ',ans1,ans2,ans3,ans4)
print(count)
