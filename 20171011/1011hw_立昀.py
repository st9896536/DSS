# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:08:19 2017

@author: user
"""
import math
import numpy as np
import statistics
from sklearn import datasets

def entropy(p1,n1):
    if(p1==0 and n1==0):
        return 1
    value = 0
    pp = p1/(p1+n1)
    pn = n1/(p1+n1)
    if(pp>0):
        value -= pp*math.log2(pp)
    if(pn>0):
        value -= pn*math.log2(pn)
    return value

def IG(p1,n1,p2,n2):
    num = p1+n1+p2+n2
    num1 = p1+n1
    num2 = p2+n2
    return entropy(p1+p2,n1+n2)-num1/num*entropy(p1,n1)-num2/num*entropy(p2,n2)

def build_tree(feature,target,test):
    node = dict()
    node['data'] = range(len(target))
    Tree = [];
    Tree.append(node)
    t = 0
    while(t<len(Tree)):
        idx = Tree[t]['data']
        if(sum(target[idx])==0):
            Tree[t]['leaf']=1
            Tree[t]['decision']=0
        elif(sum(target[idx])==len(idx)):
            Tree[t]['leaf']=1
            Tree[t]['decision']=1
        bestIG = 0
        for i in range(feature.shape[1]):
            pool = list(set(feature[idx,i]))
            for j in range(len(pool)-1):
                thres = (pool[j]+pool[j+1])/2
                G1 = []
                G2 = []
                for k in idx:
                    if(feature[k,i]<=thres):
                        G1.append(k)
                    else:
                        G2.append(k)
                thisIG = IG(sum(target[G1]==1),sum(target[G1]==0),sum(target[G2]==1),sum(target[G2]==0))
                if(thisIG>bestIG):
                    bestIG = thisIG
                    bestG1 = G1
                    bestG2 = G2
                    bestthres = thres
                    bestf = i
        if(bestIG>0):
            Tree[t]['leaf']=0
            Tree[t]['selectf']=bestf
            Tree[t]['threshold']=bestthres
            Tree[t]['child']=[len(Tree),len(Tree)+1]
            node = dict()
            node['data']=bestG1
            Tree.append(node)
            node = dict()
            node['data']=bestG2
            Tree.append(node)
        else:
            Tree[t]['leaf']=1
            if(sum(target[idx]==1)>sum(target[idx]==0)):
                Tree[t]['decision']=1
            else:
                Tree[t]['decision']=0
        t+=1
    
    test_feature = test
    now = 0
    while(Tree[now]['leaf']==0):
        bestf = Tree[now]['selectf']
        thres = Tree[now]['threshold']
        if(test_feature[bestf]<=thres):
            now = Tree[now]['child'][0]
        else:
            now = Tree[now]['child'][1]
    
    return Tree[now]['decision']
        

iris = datasets.load_iris()
data_feature=iris.data
data_target=iris.target
feature01=[]
feature02=[]
feature12=[]
target01=[]
target02=[]
target12=[]
test_ans=[]
error=0
for i in range(150):
    for j in range(150):
        if(i==j):# 將要test的資料取出
            test_feature=data_feature[i,:]
            test_target=data_target[i]
        else:##根據 target 0 1 2分成三類
            if(iris.target[j]==0):
                feature01.append(data_feature[j,:])
                feature02.append(data_feature[j,:])
                target01.append(data_target[j])
                target02.append(data_target[j])

            elif(iris.target[j]==1):
                feature01.append(data_feature[j,:])
                feature12.append(data_feature[j,:])
                target01.append(data_target[j])
                target12.append(data_target[j])

            else:
                feature02.append(data_feature[j,:])
                feature12.append(data_feature[j,:])
                target02.append(data_target[j])
                target12.append(data_target[j])
                
    new_f01=np.asarray(feature01)#list to numpy array
    new_f02=np.asarray(feature02)
    new_f12=np.asarray(feature12)
    new_t01=np.asarray(target01)
    new_t02=np.asarray(target02)
    new_t12=np.asarray(target12)
    
    a1=build_tree(new_f01,new_t01,test_feature)#build tree1
    a2=build_tree(new_f02,new_t02,test_feature)#build tree2
    a3=build_tree(new_f12,new_t12,test_feature)#build tree3
    
    test_ans.append(a1)
    test_ans.append(a2)
    test_ans.append(a3)
    
    if(a1+a2+a3==3):
        final_ans=0
    else:
        final_ans=statistics.mode(test_ans)#取眾數
    
    if(final_ans!=iris.target[i]):#判斷錯誤
        error+=1

print(error)
"""
test=[0,0,1,1,1,1,2,2,3,5,6,8]
print(statistics.mode(test)) #求眾數
print(build_tree(iris.data,iris.target))"""