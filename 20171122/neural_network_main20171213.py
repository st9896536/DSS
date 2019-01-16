# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 18:45:56 2017

@author: USER
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import io
import matplotlib.patches as patches
from PIL import Image
#將npz的資料讀進來，然後0,1,2,3的attr.分別代表：訓練臉、訓練非臉、測試臉、測試非臉
npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

trpn = trainface.shape[0]
trnn = trainnonface.shape[0]
tepn = testface.shape[0]
tenn = testnonface.shape[0]

"""
#fn代表19*19總共有幾個特徵點
fn = 0
ftable = []
#因為一張圖的大小是19*19，所以創建一個ftable大小為19*19。
#Y與X代表作標點，這個19*19的正方形的每個點
#1*2
for y in range(19):
    for x in range(19):
        #限制最小為2*2的正方形
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([0,y,x,h,w])
#2*1
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w<=19):
                    fn = fn + 1
                    ftable.append([1,y,x,h,w])
#1*3
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*3<=19):
                    fn = fn + 1
                    ftable.append([2,y,x,h,w])
#2*2
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([3,y,x,h,w])
                    
"""
#
#直接將原始圖片的點對應到19*19的哪個點，而知道這些點之後，我們就可以一次直接取他們的值
#sample是 n*361的矩陣
#fe是用來處理怎麼取特徵
#def fe2(sample,ftable,c,multiple):
#    #ftype代表是1*2，2*1，1*3還是2*2的
#    ftype = ftable[c][0]
#    y = int(ftable[c][1] * (multiple / 19))
#    x = int(ftable[c][2] * (multiple / 19))
#    h = int(ftable[c][3] * (multiple / 19))
#    w = int(ftable[c][4] * (multiple / 19))
#    #準備一個361的矩陣，再把它reshape，所以他會變成first row 0 1 2 .....18 
#    #second row 19 20 21 .....37，然後再靠著y,x,h,w取出對應的方塊，最後到一維的sample取值
##    T = np.arange(multiple*multiple).reshape((multiple,multiple))
#    if(ftype==0):
#        #T[y:y+h,x:x+w].flatten()是把該範圍的位置是19*19的哪個區域取出來，拉成一維的，再到原圖(也就是1*361)找取出位置的值(長方形)
#        #將每一row sum起來，之後再把全部加起來，所以會有sample.shape[0]個值
#        output = np.sum(sample[y:y+h,x:x+w])-np.sum(sample[y:y+h,x+w:x+w*2])
#    if(ftype==1):
#        output = -np.sum(sample[y:y+h,x:x+w])+np.sum(sample[y+h:y+h*2,x:x+w])
#    if(ftype==2):
#        output = np.sum(sample[y:y+h,x:x+w])-np.sum(sample[y:y+h,x+w:x+w*2])+np.sum(sample[y:y+h,x+w*2:x+w*3])
#    if(ftype==3):
#        output = np.sum(sample[y:y+h,x:x+w])-np.sum(sample[y:y+h,x+w:x+w*2])-np.sum(sample[y+h:y+h*2,x:x+w])+np.sum(sample[y+h:y+h*2,x+w:x+w*2])
#    #return的output會是一個一維陣列
#    return output

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
#    return (integral[(y+1+h-1),(x+1+w-1)]+integral[y+1,x+1]) - (integral[y+1,x+1+w-1] + integral[(y+1+h-1),x+1]
    return integral[y+h,x+w]+integral[y,x]-integral[y,x+w]-integral[y+h,x]
           
def fe_integral(sample,ftable,c,multiple,integral):
    ftype = ftable[c][0]
    y = int(ftable[c][1] * (multiple / 19))
    x = int(ftable[c][2] * (multiple / 19))
    h = int(ftable[c][3] * (multiple / 19))
    w = int(ftable[c][4] * (multiple / 19))
    if(ftype==0):
        output = cal_area(sample,y,h,x,w)-cal_area(sample,y,h,x+w,w)
    if(ftype==1):
        output = -cal_area(sample,y,h,x,w)+cal_area(sample,y+h,h,x,w)
    if(ftype==2):
        output = cal_area(sample,y,h,x,w)-cal_area(sample,y,h,x+w,w)+cal_area(sample,y,h,(x+2*w),w)
    if(ftype==3):
        output = cal_area(sample,y,h,x,w)-cal_area(sample,y,h,x+w,w)-cal_area(sample,y+h,h,x,w)+cal_area(sample,y+h,h,x+w,w)
    return output
#def fe(sample,ftable,c):
#    #ftype代表是1*2，2*1，1*3還是2*2的
#    ftype = ftable[c][0]
#    y = ftable[c][1]
#    x = ftable[c][2]
#    h = ftable[c][3]
#    w = ftable[c][4]
#    #準備一個361的矩陣，再把它reshape，所以他會變成first row 0 1 2 .....18 
#    #second row 19 20 21 .....37，然後再靠著y,x,h,w取出對應的方塊，最後到一維的sample取值
#    T = np.arange(361).reshape((19,19))
#    if(ftype==0):
#        #T[y:y+h,x:x+w].flatten()是把該範圍的位置是19*19的哪個區域取出來，拉成一維的，再到原圖(也就是1*361)找取出位置的值(長方形)
#        #將每一row sum起來，之後再把全部加起來，所以會有sample.shape[0]個值
#        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)
#    if(ftype==1):
#        output = -np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)+np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis=1)
#    if(ftype==2):
#        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)+np.sum(sample[:,T[y:y+h,x+w*2:x+w*3].flatten()],axis=1)
#    if(ftype==3):
#        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)-np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis=1)+np.sum(sample[:,T[y+h:y+h*2,x+w:x+w*2].flatten()],axis=1)
#    #return的output會是一個一維陣列
#    return output


trpf = np.zeros((trpn,fn))
trnf = np.zeros((trnn,fn))
#總共做36648次
#for c in range(fn):
#    trpf[:,c] = fe(trainface,ftable,c)
#    trnf[:,c] = fe(trainnonface,ftable,c)
    

#positive weight
#除以自己是因為trpn和trnn的個數不一樣，所以為了公平，就除以自己的個數
pw = np.ones((trpn,1))/trpn/2
#negative weight
nw = np.ones((trnn,1))/trnn/2

#找個個特徵點切一刀，計算比他大還是比她小
def WC(pw,nw,pf,nf):
    #找最大的positive feature，最大的negative feature，然後再從這兩個數值裡面找最大的
    maxf = max(pf.max(),nf.max())
    #找最小的positive feature，最小的negative feature，然後再從這兩個數值裡面找最小的
    minf = min(pf.min(),nf.min())
    #(數線的概念很好理解)找最大值和最小值的目的是給定範圍，然後從這範圍裏面切十刀，看誰最好 
    #這是第一刀
    theta = (maxf-minf)/10+minf
    #把pf<theta的人的pw值加起來，然後把nf>=theta的nw加起來，最後再加起來，這就是全部的錯誤
    error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
    #是否有交換
    polarity = 1
    #如果錯誤超過一半
    if(error>0.5):
        #那polarity就設成有交換，然後錯誤值就是1-error，因為已經把原本的東西交換過了
        error = 1-error
        polarity = 0
    min_theta = theta
    min_error = error
    min_polarity = polarity
    for i in range(2,10):
        theta = (maxf-minf)*i/10+minf
        error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
        polarity = 1
        if(error>0.5):
            error = 1-error
            polarity = 0
        if(error<min_error):
            min_theta = theta
            min_error = error
            min_polarity = polarity
    return min_error,min_theta,min_polarity

SC = []
for t in range(20):
    weightsum = np.sum(pw)+np.sum(nw)
    pw = pw/weightsum
    nw = nw/weightsum
    #先有trpf[:,0]的best_error,best_theta,_best_polarity
    best_error,best_theta,best_polarity = WC(pw,nw,trpf[:,0],trnf[:,0])
    best_feature = 0
    for i in range(1,fn):
        error,theta,polarity = WC(pw,nw,trpf[:,i],trnf[:,i])
        if(error<best_error):
            best_feature = i
            best_error = error
            best_theta = theta
            best_polarity = polarity
    beta = best_error/(1-best_error)
    alpha = math.log10(1/beta)
    SC.append([best_feature,best_theta,best_polarity,alpha])
    #分對的人，權重乘上beta，分錯的人，權重不變
    #polarity == 1代表沒有改變
    if(best_polarity==1):
        pw[trpf[:,best_feature]>=best_theta] = pw[trpf[:,best_feature]>=best_theta]*beta
        nw[trnf[:,best_feature]<best_theta] = nw[trnf[:,best_feature]<best_theta]*beta
    #代表有改變
    else:
        pw[trpf[:,best_feature]<best_theta] = pw[trpf[:,best_feature]<best_theta]*beta
        nw[trnf[:,best_feature]>=best_theta] = nw[trnf[:,best_feature]>=best_theta]*beta
    
sc = SC
#
#trps = np.zeros((trpn,1))
#trns = np.zeros((trnn,1))
#alpha_sum = 0
#for i in range(20):
#    feature = SC[i][0]
#    theta = SC[i][1]
#    polarity = SC[i][2]
#    alpha = SC[i][3]
#    alpha_sum = alpha_sum + alpha
#    if(polarity==1):
#        trps[trpf[:,feature]>=theta] = trps[trpf[:,feature]>=theta]+alpha
#        trns[trnf[:,feature]>=theta] = trns[trnf[:,feature]>=theta]+alpha
#    else:
#        trps[trpf[:,feature]<theta] = trps[trpf[:,feature]<theta]+alpha
#        trns[trnf[:,feature]<theta] = trns[trnf[:,feature]<theta]+alpha
#
#trps = trps/alpha_sum
#trns = trns/alpha_sum

#test = trainface[0].reshape((1,361))
#trps = np.zeros((1,1))
#testpf = np.zeros((1,fn))
#for c in range(fn):
#    testpf[:,c] = fe(test,ftable,c)
#alpha_sum = 0
#for i in range(20):
#    feature = sc[i][0]
#    theta = sc[i][1]
#    polarity = sc[i][2]
#    alpha = sc[i][3]
#    alpha_sum = alpha_sum + alpha
#    if(polarity==1):
#        trps[testpf[:,feature]>=theta] = trps[testpf[:,feature]>=theta]+alpha
#    else:
#        trps[testpf[:,feature]<theta] = trps[testpf[:,feature]<theta]+alpha


#trps = trps/alpha_sum
#print(trps)
#trps = trps/alpha_sum

#test = io.imread('D:\\Intellegence\\lesson9\\face00001.pgm')
#test = np.array(test,dtype = 'f')
#resultpicture = test.reshape((19,19))
#integral_area = cal_integral(test)
##test = io.imread('D:\\Intellegence\\lesson9\\test3838.pgm')
##test = np.array(test,dtype = 'f')
##integral_area = cal_integral(test)
##resultpicture = test.reshape((38,38))
######whole_picture_block = np.arange(1444).reshape((38,38))
####resultpicture = test.reshape((38,38))
#for k in range(19,20,1):
#    for i in range(20-k):
#        for j in range(20-k):
#            trps = 0.0
##            selected_block = test[i:(k*1+i),j:(k*1+j)]
#            selected_block = integral_area[i:((k*1)+(i+1)),j:(k*1+(j+1))]
#            print('i: ' + str(i) +'  '+ 'j: ' + str(j))
#            alpha_sum = 0
#            y,x,h,w = 0,0,0,0
#            for t in range(20):
#                feature = sc[t][0]
#                theta = sc[t][1]*((k/19)**2)
#                polarity = sc[t][2]
#                alpha = sc[t][3]
#                alpha_sum = alpha_sum + alpha
#                feature_value = fe_integral(selected_block,ftable,feature,k,integral_area)
#                print(feature_value)
#                if(polarity==1):
#                    if feature_value>=theta:
#                        trps += alpha
#                else:
#                    if feature_value<theta:
#                        trps += alpha
#            trps = trps/alpha_sum
#            print(trps)
#            if(trps>0.5):
#                plt.imshow(resultpicture)
#                currentAxis=plt.gca()
#                rect=patches.Rectangle((i, j),k,k,linewidth=1,edgecolor='r',facecolor='none')
#                currentAxis.add_patch(rect)

picture = io.imread('/Users/apple/Desktop/happy-people-friends.jpg',as_grey=True) #轉灰階
picture = picture*255
resultpicture = io.imread('/Users/apple/Desktop/happy-people-friends.jpg')
integral_area = cal_integral(picture)
for k in range(19,58,19):
    for i in range(750-k):
        for j in range(1024-k):
            trps = 0.0
#            selected_block = picture[i:(k*1+i),j:(k*1+j)]
            selected_block = integral_area[i:((k*1)+(i+1)),j:(k*1+(j+1))]
            alpha_sum = 0
            y,x,h,w = 0,0,0,0
            for t in range(20):
                feature = sc[t][0]
                theta = sc[t][1]*((k/19)**2)
                polarity = sc[t][2]
                alpha = sc[t][3]
                alpha_sum = alpha_sum + alpha
                feature_value = fe_integral(selected_block,ftable,feature,k,integral_area)
                if(polarity==1):
                    if feature_value>=theta:
                        trps += alpha
                else:
                    if feature_value<theta:
                        trps += alpha
            trps = trps/alpha_sum
            if(trps>0.65):
                print('i: ' + str(i) +'  '+ 'j: ' + str(j) + 'k: ' + str(k))
                plt.imshow(resultpicture)
                currentAxis=plt.gca()
                if k == 19:
                    rect=patches.Rectangle((j, i),k,k,linewidth=1,edgecolor='r',facecolor='none')
                    currentAxis.add_patch(rect)
                elif k == 38:
                    rect=patches.Rectangle((j, i),k,k,linewidth=1,edgecolor='g',facecolor='none')
                    currentAxis.add_patch(rect)
                else:
                    rect=patches.Rectangle((j, i),k,k,linewidth=1,edgecolor='b',facecolor='none')
                    currentAxis.add_patch(rect)
                plt.savefig("testfig.png")
                print(trps)










