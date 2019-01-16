#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:57:37 2017

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
#策略0.0
profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    profit[i] = TAIEX[idx[-1],1] - TAIEX[idx[0],2]
profit2 = np.cumsum(profit)
plt.plot(profit2)
plt.show()

ans1 = profit2[-1] #總損益點數士profit2的最後一筆數值
ans2 = np.sum(profit>0)/len(profit) #勝率 profit>0的個數除以總個數
ans3 = np.mean(profit[profit>0]) #賺錢時平均每次獲利點數
ans4 = np.mean(profit[profit<=0]) #輸錢時平均每次獲利點數
plt.hist(profit,bins=100) #繪出每日損益的分布圖
plt.show()

print('0.0: ',ans1,ans2,ans3,ans4)

#----------------------------------------------
#策略0.1
profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    profit[i] = -(TAIEX[idx[-1],1] - TAIEX[idx[0],2])
profit2 = np.cumsum(profit)
plt.plot(profit2)
plt.show()


ans1 = profit2[-1] #總損益點數士profit2的最後一筆數值
ans2 = np.sum(profit>0)/len(profit) #勝率 profit>0的個數除以總個數
ans3 = np.mean(profit[profit>0]) #賺錢時平均每次獲利點數
ans4 = np.mean(profit[profit<=0]) #輸錢時平均每次獲利點數
plt.hist(profit,bins=100) #繪出每日損益的分布圖

plt.show()

print('0.1: ',ans1,ans2,ans3,ans4)

#----------------------------------------------
#策略1.0
#開盤買進一口，30點停損，收盤平倉
profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0] 
    idx.sort()
    
    p1 = TAIEX[idx[0],2] #一開盤的第一分鐘開盤價
    idx2 = np.nonzero(TAIEX[idx,4]<=p1-30)[0] #停損
    if(len(idx2)==0): #那天沒撞到停損價
        p2 = TAIEX[idx[-1],1] #如果沒有的話 以當天的收盤價收盤
    else: #那天有撞到的話
        p2 = TAIEX[idx[idx2[0]],1]  #找出那天的停損價
    profit[i] = p2-p1  #ans1:948.0
    
profit2 = np.cumsum(profit) #累加第一天到最後一天的profit 放到profit2裡面
plt.plot(profit2)
plt.show()

ans1 = profit2[-1] #總損益點數士profit2的最後一筆數值
ans2 = np.sum(profit>0)/len(profit) #勝率 profit>0的個數除以總個數
ans3 = np.mean(profit[profit>0]) #賺錢時平均每次獲利點數
ans4 = np.mean(profit[profit<=0]) #輸錢時平均每次獲利點數
plt.hist(profit,bins=100) #繪出每日損益的分布圖
plt.show()

print('1.0: ',ans1,ans2,ans3,ans4)

    
#----------------------------------------------
#策略1.1
#開盤空一口，30點停損，收盤平倉
#停損要往上看 越漲我還的錢要越多 要看3 <=30 改 >=30 然後改成負號

profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0] 
    idx.sort()
    
    p1 = TAIEX[idx[0],2] #一開盤的第一分鐘開盤價
    idx2 = np.nonzero(TAIEX[idx,3]>=p1+30)[0] #停損
    if(len(idx2)==0): #那天沒撞到停損價
        p2 = TAIEX[idx[-1],1] #如果沒有的話 以當天的收盤價收盤
    else: #那天有撞到的話
        p2 = TAIEX[idx[idx2[0]],1]  #找出那天的停損價
    profit[i] = -(p2-p1) 
    
profit2 = np.cumsum(profit) #累加第一天到最後一天的profit 放到profit2裡面
plt.plot(profit2)
plt.show()
 
ans1 = profit2[-1] #總損益點數士profit2的最後一筆數值
ans2 = np.sum(profit>0)/len(profit) #勝率 profit>0的個數除以總個數
ans3 = np.mean(profit[profit>0]) #賺錢時平均每次獲利點數
ans4 = np.mean(profit[profit<=0]) #輸錢時平均每次獲利點數
plt.hist(profit,bins=100) #繪出每日損益的分布圖
plt.show()

print('1.1: ',ans1,ans2,ans3,ans4)

#----------------------------------------------
#策略2.0
#開盤買進一口，30點停損，30點停利，收盤平倉    
profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0] 
    idx.sort()
    p1 = TAIEX[idx[0],2] #一開盤的第一分鐘開盤價
    idx2 = np.nonzero(TAIEX[idx,4]<=p1-30)[0] #停損 
    idx3 = np.nonzero(TAIEX[idx,3]>=p1+30)[0] #停利
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
    profit[i] = p2-p1
    
profit2 = np.cumsum(profit) #累加第一天到最後一天的profit 放到profit2裡面
plt.plot(profit2)
plt.show()

ans1 = profit2[-1] #總損益點數士profit2的最後一筆數值
ans2 = np.sum(profit>0)/len(profit) #勝率 profit>0的個數除以總個數
ans3 = np.mean(profit[profit>0]) #賺錢時平均每次獲利點數
ans4 = np.mean(profit[profit<=0]) #輸錢時平均每次獲利點數
plt.hist(profit,bins=100) #繪出每日損益的分布圖
plt.show()

print('2.0: ',ans1,ans2,ans3,ans4)

#----------------------------------------------
#策略2.1
#開盤空一口，30點停損，30點停利，收盤平倉
profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0] 
    idx.sort()
    p1 = TAIEX[idx[0],2] #一開盤的第一分鐘開盤價
    idx2 = np.nonzero(TAIEX[idx,3]>=p1+30)[0] #停利 
    idx3 = np.nonzero(TAIEX[idx,4]<=p1-30)[0] #停損
    if len(idx2)==0 and len(idx3)==0: #如果沒有停利跟停損
        p2 = TAIEX[idx[-1],1] #如果沒有的話 以當天的收盤價收盤
    elif len(idx3) == 0: #停利
        p2 = TAIEX[idx[idx2[0]],1]  
    elif len(idx2) == 0: #停損
        p2 = TAIEX[idx[idx3[0]],1]
    elif idx2[0]<idx3[0]:  #停損比停利早出現
        p2 = TAIEX[idx[idx2[0]],1]
    else:
        p2 = TAIEX[idx[idx3[0]],1]
    profit[i] = -(p2-p1)
    
profit2 = np.cumsum(profit) #累加第一天到最後一天的profit 放到profit2裡面
plt.plot(profit2)
plt.show()

ans1 = profit2[-1] #總損益點數士profit2的最後一筆數值
ans2 = np.sum(profit>0)/len(profit) #勝率 profit>0的個數除以總個數
ans3 = np.mean(profit[profit>0]) #賺錢時平均每次獲利點數
ans4 = np.mean(profit[profit<=0]) #輸錢時平均每次獲利點數
plt.hist(profit,bins=100) #繪出每日損益的分布圖
plt.show()

print('2.1: ',ans1,ans2,ans3,ans4)

