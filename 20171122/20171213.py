#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 18:44:32 2017

@author: Rong
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

trpn = trainface.shape[0]
trnn = trainnonface.shape[0]
tepn = testface.shape[0]
tenn = testnonface.shape[0] #四種data的個數

fn = 0 #feature 36648個值可以來描述我這張照片
ftable = [] 
for y in range(19):
    for x in range(19):
        for h in range(2,20): #min 2 max 19 check右下角那個點還在圖裡面就合法
            for w in range(2,20):  #右下角座標 y:y+h-1 x:x+2w-1
                if(y+h<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([0,y,x,h,w]) #每一個合法特徵都把他留下來
                    
for y in range(19):
    for x in range(19):
        for h in range(2,20): #min 2 max 19 check右下角那個點還在圖裡面就合法
            for w in range(2,20):  #右下角座標 y:y+2h-1 x:x+w-1
                if(y+h*2<=19 and x+w<=19):
                    fn = fn + 1
                    ftable.append([1,y,x,h,w]) #第一個值為幾號特徵
                    
for y in range(19):
    for x in range(19):
        for h in range(2,20): #min 2 max 19 check右下角那個點還在圖裡面就合法
            for w in range(2,20):  #右下角座標 y:y+h-1 x:x+3w-1
                if(y+h<=19 and x+w*3<=19):
                    fn = fn + 1
                    ftable.append([2,y,x,h,w]) 
                    
for y in range(19):
    for x in range(19):
        for h in range(2,20): #min 2 max 19 check右下角那個點還在圖裡面就合法
            for w in range(2,20):  #右下角座標 y:y+2h-1 x:x+2w-1
                if(y+h*2<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([3,y,x,h,w]) #36648 * 5 的 array
                    
def cal_integral(sample):
    rownumber = sample.shape[0]+1
    columnnumber = sample.shape[1]+1
    integral = np.zeros((rownumber,columnnumber))
    for i in range(1,rownumber):
        for j in range(1,columnnumber):
            integral[i,j] = np.sum(sample[0:i,0:j])
    #return的output會是一個一維陣列
    return integral   

def cal_area(integral,y,h,x,w):
#   return (integral[(y+1+h-1),(x+1+w-1)]+integral[y+1,x+1]) - (integral[y+1,x+1+w-1] + integral[(y+1+h-1),x+1]
    return integral[y+h,x+w]+integral[y,x]-integral[y,x+w]-integral[y+h,x]
   

                 


#return a vector with N feature values                   
def fe(sample,ftable,c): #sample 為 n*361的矩陣 fe:feature extraction
    ftype = ftable[c][0]
    y = ftable[c][1]
    x = ftable[c][2]
    h = ftable[c][3]
    w = ftable[c][4]
    T = np.arange(361).reshape((19,19))  #矩陣類型運算小技巧 用lookup的方式去處理照片
    if ftype==0:
        #白色區域的座標 - 黑色區域的座標 把照片的這些column取出來 sum 擺在右邊 
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1) - np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)  
    if ftype==1:
        #白色區域的座標 把照片的這些column取出來 sum 擺在右邊 
        output = -np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1) + np.sum(sample[:,T[y:y+h*2,x:x+w].flatten()],axis=1)  
    if ftype==2: #白黑白
        #白色區域的座標 - 黑色區域的座標 把照片的這些column取出來 sum 擺在右邊 
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1) - np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1) + np.sum(sample[:,T[y:y+h*2,x:x+w*3].flatten()],axis=1)  
    if ftype==3: #白黑黑白
        #白色區域的座標 - 黑色區域的座標 把照片的這些column取出來 sum 擺在右邊 
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1) - np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1) - np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis=1) + np.sum(sample[:,T[y+h:y+h*2,x+w:x+w*2].flatten()],axis=1)  
    
    
    return output
    

trpf = np.zeros((trpn,fn))
trnf = np.zeros((trnn,fn))
#tepf = np.zeros((tepn,fn))
#tenf = np.zeros((tenn,fn)) #開四個array

for c in range(fn):
    trpf[:,c] = fe(trainface,ftable,c)
    trnf[:,c] = fe(trainnonface,ftable,c)
    #tepf[:,c] = fe(testface,ftable,c)
    #tenf[:,c] = fe(testnonface,ftable,c)

#adaboost 演算法
pw = np.ones((trpn,1))/trpn/2  #positive weight 
nw = np.ones((trnn,1))/trnn/2


def WC(pw,nw,pf,nf): #WC: weak classification
    maxf = max(pf.max(),nf.max())
    minf = min(pf.min(),nf.min())
    theta = (maxf - minf) / 10 + minf #第一刀 左邊數來1/10
    error = np.sum(pw[pf<theta]) + np.sum(nw[nf>=theta]) 
    polarity = 1
    if error > 0.5:
        error = 1 - error
        polarity = 0
    min_theta = theta
    min_error = error
    min_polarity = polarity
    for i in range(2,10):
        theta = (maxf - minf) * i / 10 + minf 
        error = np.sum(pw[pf<theta]) + np.sum(nw[nf>=theta]) 
        polarity = 1
        if error > 0.5:
            error = 1 - error
            polarity = 0
        if error < min_error:
            min_theta = theta
            min_error = error
            min_polarity = polarity
    return min_error,min_theta,min_polarity

SC = []
for t in range(10):
    weightsum = np.sum(pw) + np.sum(nw)
    pw = pw/weightsum
    nw = nw/weightsum
    best_error,best_theta,best_polarity = WC(pw,nw,trpf[:,0],trnf[:,0])
    best_feature = 0
    for i in range(1,fn):
        error,theta,polarity = WC(pw,nw,trpf[:,0],trnf[:,0])
        if(error<best_error):
            best_feature = i
            best_error = error
            best_theta = theta
            best_polarity = polarity

    beta = best_error/(1-best_error)
    alpha = math.log10(1/beta)
    SC.append([best_feature,best_theta,best_polarity,alpha])
    #被分隊的人權重要縮小
    if best_polarity == 1: #右正左負
        pw[trpf[:,best_feature]>=best_theta] = pw[trpf[:,best_feature]>=best_theta] * beta #被分對的人
        nw[trnf[:,best_feature]<best_theta] = nw[trnf[:,best_feature]<best_theta] * beta
    else:
        pw[trpf[:,best_feature]<best_theta] = pw[trpf[:,best_feature]<best_theta] * beta 
        nw[trnf[:,best_feature]>=best_theta] = nw[trnf[:,best_feature]>=best_theta] * beta
        
        
trps = np.zeros((trpn,1))
trns = np.zeros((trnn,1))
alpha_sum = 0

for i in range(10): #n個特徵
    feature = SC[i][0]
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum = alpha_sum + alpha
    if polarity == 1:
        trps[trpf[:,feature] >= theta] = trps[trpf[:,feature] >= theta] + alpha
        trns[trnf[:,feature] >= theta] = trns[trnf[:,feature] >= theta] + alpha
    else:
        trps[trpf[:,feature] < theta] = trps[trpf[:,feature] < theta] + alpha
        trns[trnf[:,feature] < theta] = trns[trnf[:,feature] < theta] + alpha

trps = trps/alpha_sum
trns = trns/alpha_sum
    
    
#Load picture
im = Image.open("26057855_2105735862987494_1865787697_n.jpg")
im2 = np.asarray(im)
im2=np.array(im.convert('L')).astype(int) #轉灰階

#feature extration from dicts to list 
H=im2.shape[0] 
W=im2.shape[1]
total=[]
out=[]
feature=0
for i in range(0,H,2): #圖片的高
    for j in range(0,W,2): #圖片的寬
        if(i+19<=H and j+19<=W):
            img = im2[i:i+19,j:j+19].flatten()
            img = img.tolist()
            total.append(img) #N * 361
            out.append([j,i]) #特徵座標
            feature += 1 #N總數
total = np.asarray(total) #N*361個特徵put in list

#計算分數
trps2 = np.zeros((feature,1))
#alpha_sum = 0

for i in range(len(SC)):
    output=fe(total,ftable,SC[i][0])
    feature = SC[i][0]
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    #alpha_sum = alpha_sum + alpha
    if(polarity==1):
        trps2[output>=theta] = trps2[output>=theta]+alpha
    else:
        trps2[output<theta] = trps2[output<theta]+alpha
trps2 = trps2/alpha_sum

#對應位置
located=[]
for i in range(feature):
    if(trps2[i]>0.75):
        located.append(out[i])


#plot the photo
fig,ax=plt.subplots(1)
ax.imshow(im)
for i in range(len(located)):
    result=patches.Rectangle((located[i][0],located[i][1]),19,19,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(result)
    
plt.show()
fig.savefig('face_capture.png')    
    
    
