#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:18:52 2017

@author: Rong
"""

import math
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()


def entropy(p1,n1):
    if(p1 == 0 and n1 == 0):
        return 1
    value = 0
    pp = p1/(p1+n1)
    pn = n1/(p1+n1)
    if(pp>0):
        value -= pp * math.log2(pp)
    if(pn>0):
        value -= pn * math.log2(pn)
    return value

def IG(p1,n1,p2,n2):
    num = p1+n1+p2+n2
    num1 = p1+n1
    num2 = p2+n2
    return entropy(p1+p2,n1+n2)-num1/num * entropy(p1,n1) - num2/num * entropy(p2,n2)

#data = np.loadtxt('PlayTennis.txt',usecols=range(5),dtype=int)

feature = iris.data 
target = iris.target  # 0 1 2



#第一棵樹 01
def decisiontree1(testindex,Tree):
    #判斷testindex 有沒有在tmp 裡面
    node = dict()
    tmp = list(range(100))
    if testindex in tmp:
        tmp.remove(testindex)
    node['data'] = tmp
    Tree.append(node)
        
    
    t = 0   #先處理第0節點    
    while (t<len(Tree)):  #檢查i有沒有走道樹底 走到底就不走了！
        idx = Tree[t]['data']
        if(sum(target[idx])==0): #如果node[i]裡面全部人都不打球
            Tree[t]['leaf'] = 1 #1為葉節點
            Tree[t]['decision'] = 0 
        elif(sum(target[idx])==len(idx)): #如果全部都打球
            Tree[t]['leaf'] = 1
            Tree[t]['decision'] = 1       
        bestIG = 0
        # 把所有的特徵都跑完
        for i in range(feature.shape[1]): #所以i會跑所有的column 4個特徵 i是所有特徵
            #就會拿到這5筆的特徵值 0.2 0.4 0.6 0.4 0.2 這個池子
            pool = list(set(feature[idx,i]))  #[把重複的數值篩掉 取 0.2 0.4 0.6
            pool.sort()
            for j in range(len(pool)-1): #pool 有3個 我只要跑2個 j是刀
                thres = (pool[j]+pool[j+1])/2 #取中位數 0.2 0.4 取0.3   0.4 0.6 取 0.5
                G1 = []
                G2 = []
                for k in idx:
                    if(feature[k,i]<=thres):
                        G1.append(k) #把k加到左子樹裡面
                    else:
                        G2.append(k)
                    #分成兩個group sum去算裡面有幾個0 幾個1
                thisIG = IG(sum(target[G1]==1),sum(target[G1]==0),sum(target[G2]==1),sum(target[G2]==0))
                if(thisIG>bestIG):
                    #如果這次的IG比bestIG好 更新資料
                    bestIG = thisIG
                    bestG1 = G1
                    bestG2 = G2
                    bestthres = thres
                    bestf = i #最好的特徵
        if(bestIG>0):  #檢查最好的切法有沒有用
            Tree[t]['leaf'] = 0
            Tree[t]['selectf'] = bestf
            Tree[t]['threshold'] = bestthres
            Tree[t]['child']=[len(Tree),len(Tree)+1] #兩個左右子節點 放在tree的尾巴
            node = dict()
            node['data'] = bestG1 
            Tree.append(node)
            node = dict()
            node['data'] = bestG2
            Tree.append(node)
        else:  #如果最好的切法都沒有用
            Tree[t]['leaf'] = 1
            if sum(target[idx] == 1) > sum(target[idx] == 0):
                Tree[t]['decision'] = 1 #採多數決 如果1多 就投1
            else:
                Tree[t]['decision'] = 0
        t+=1        


#第二棵樹 12
def decisiontree2(testindex,Tree):
    node = dict()
    tmp = list(range(50,150))
    if testindex in tmp:
        tmp.remove(testindex)
    node['data'] = tmp
    Tree.append(node)
        
        
    t = 0   #先處理第0節點    
    while (t<len(Tree)):  #檢查i有沒有走道樹底 走到底就不走了！
        idx = Tree[t]['data']
        if(sum(target[idx])==1): #如果node[i]裡面全部人都不打球
            Tree[t]['leaf'] = 1 #1為葉節點
            Tree[t]['decision'] = 1 
        elif(sum(target[idx])==len(idx)): #如果全部都打球
            Tree[t]['leaf'] = 1
            Tree[t]['decision'] = 2   
            
        bestIG = 0
        # 把所有的特徵都跑完
        for i in range(feature.shape[1]): #所以i會跑所有的column 4個特徵 i是所有特徵
            #就會拿到這5筆的特徵值 0.2 0.4 0.6 0.4 0.2 這個池子
            pool = list(set(feature[idx,i]))  #[把重複的數值篩掉 取 0.2 0.4 0.6
            pool.sort()
            for j in range(len(pool)-1): #pool 有3個 我只要跑2個 j是刀
                thres = (pool[j]+pool[j+1])/2 #取中位數 0.2 0.4 取0.3   0.4 0.6 取 0.5
                G1 = []
                G2 = []
                for k in idx:
                    if(feature[k,i]<=thres):
                        G1.append(k) #把k加到左子樹裡面
                    else:
                        G2.append(k)
                #分成兩個group sum去算裡面有幾個0 幾個1
                thisIG = IG(sum(target[G1]==2),sum(target[G1]==1),sum(target[G2]==2),sum(target[G2]==1))
                if(thisIG>bestIG):
                    #如果這次的IG比bestIG好 更新資料
                    bestIG = thisIG
                    bestG1 = G1
                    bestG2 = G2
                    bestthres = thres
                    bestf = i #最好的特徵
        if(bestIG>0):  #檢查最好的切法有沒有用
            Tree[t]['leaf'] = 0
            Tree[t]['selectf'] = bestf
            Tree[t]['threshold'] = bestthres
            Tree[t]['child']=[len(Tree),len(Tree)+1] #兩個左右子節點 放在tree的尾巴
            node = dict()
            node['data'] = bestG1 
            Tree.append(node)
            node = dict()
            node['data'] = bestG2
            Tree.append(node)
        else:  #如果最好的切法都沒有用
            Tree[t]['leaf'] = 1
            if sum(target[idx] == 1) > sum(target[idx] == 2):
                Tree[t]['decision'] = 1 #採多數決 如果1多 就投1
            else:
                Tree[t]['decision'] = 2
        t+=1        
    

#第三棵樹 02
def decisiontree3(testindex,Tree):
    node = dict()
    tmp = list(range(50))+list(range(100,150))
    if testindex in tmp:
        tmp.remove(testindex)
    node['data'] = tmp
    Tree.append(node)
    
    t = 0   #先處理第0節點    
    while (t<len(Tree)):  #檢查i有沒有走道樹底 走到底就不走了！
        idx = Tree[t]['data']
        if(sum(target[idx])==0): #如果node[i]裡面全部人都不打球
            Tree[t]['leaf'] = 1 #1為葉節點
            Tree[t]['decision'] = 0 
        elif(sum(target[idx])==len(idx)): #如果全部都打球
            Tree[t]['leaf'] = 1
            Tree[t]['decision'] = 2
       
        bestIG = 0
        # 把所有的特徵都跑完
        for i in range(feature.shape[1]): #所以i會跑所有的column 4個特徵 i是所有特徵
            #就會拿到這5筆的特徵值 0.2 0.4 0.6 0.4 0.2 這個池子
            pool = list(set(feature[idx,i]))  #[把重複的數值篩掉 取 0.2 0.4 0.6
            pool.sort()
            for j in range(len(pool)-1): #pool 有3個 我只要跑2個 j是刀
                thres = (pool[j]+pool[j+1])/2 #取中位數 0.2 0.4 取0.3   0.4 0.6 取 0.5
                G1 = []
                G2 = []
                for k in idx:
                    if(feature[k,i]<=thres):
                        G1.append(k) #把k加到左子樹裡面
                    else:
                        G2.append(k)
                #分成兩個group sum去算裡面有幾個0 幾個1
                thisIG = IG(sum(target[G1]==0),sum(target[G1]==2),sum(target[G2]==0),sum(target[G2]==2))
                if(thisIG>bestIG):
                    #如果這次的IG比bestIG好 更新資料
                    bestIG = thisIG
                    bestG1 = G1
                    bestG2 = G2
                    bestthres = thres
                    bestf = i #最好的特徵
        if bestIG>0:  #檢查最好的切法有沒有用
           Tree[t]['leaf'] = 0
           Tree[t]['selectf'] = bestf
           Tree[t]['threshold'] = bestthres
           Tree[t]['child']=[len(Tree),len(Tree)+1] #兩個左右子節點 放在tree的尾巴
           node = dict()
           node['data'] = bestG1 
           Tree.append(node)
           node = dict()
           node['data'] = bestG2
           Tree.append(node)
        else:  #如果最好的切法都沒有用
            Tree[t]['leaf'] = 1
            if sum(target[idx] == 0) > sum(target[idx] == 2):
                Tree[t]['decision'] = 0 #採多數決 如果2多 就投2
            else:
                Tree[t]['decision'] = 2
        t+=1        
    


def test(Tree,i):
    test_feature = feature[i,:] #第i筆資料拿來當測試
    now = 0
    while(Tree[now]['leaf']==0): #當還沒有到葉節點的話就繼續跑
        bestf = Tree[now]['selectf']
        thres = Tree[now]['threshold']
        if(test_feature[bestf]<=thres):
            now = Tree[now]['child'][0]
        else:
            now = Tree[now]['child'][1]  
    return Tree[now]['decision']
    
            

count = 0  #計算與實際值不同的次數
for a in range(len(target)):
    
    Tree1 = []
    Tree2 = []
    Tree3 = []   
    
    decisiontree1(a,Tree1) #把每一筆測試資料拿去decisiontree
    decisiontree2(a,Tree2) #把每一筆測試資料拿去decisiontree
    decisiontree3(a,Tree3) #把每一筆測試資料拿去decisiontree

    arrpredict = [] #拿來存三棵樹測試出來的結果
    
    arrpredict.append(test(Tree1,a))
    arrpredict.append(test(Tree2,a))
    arrpredict.append(test(Tree3,a))
    
    #countarr 是一個array 存 各個值出現幾次
    countarr = np.bincount(arrpredict)  #去找出list裡面出現最多的 
    predictnum = np.argmax(countarr)  # 如果是0 0 1 就會取出1
    
    #判斷如果投票結果 0 1 2 各出現1票 predictnum 就投0
    if countarr[predictnum] == 1:
        predictnum = 0    
    
    #計算次數
    if target[a] != predictnum: #如果預測結果跟實際不一樣
        count+=1
print(count)
    
    
    
        


    

