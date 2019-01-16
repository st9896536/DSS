#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:35:22 2017

@author: Rong
"""


from PIL import Image
from scipy import signal
import numpy as np


def conv2(I,M): #做filtering
    IH,IW = I.shape
    MH,MW = M.shape
    out = np.zeros((IH-MH+1,IW-MW+1)) #新的長跟寬（矩陣）
    for h in range(IH-MH+1): #用兩個for迴圈跑大張原圖
        for w in range(IW-MW+1):
            for y in range(MH): #用y跟x去跑mask
                for x in range(MW): #h,w目前座標 M:權重
                    out[h,w] = out[h,w] + I[h+y,w+x]*M[y,x]
    return out
    

I = Image.open('qZqZo6SVk6ecq6Q.jpg')
I.show()

data = np.asarray(I)
data2 = np.zeros((640,640,3)).astype('uint8')
#data2[:,:,1] = data[:,:,1]
#data2 = 255 - data

#------------------作業-----------------

#---------------第二題------------------motion filter

M = np.ones((1,20))/20
R = data[:,:,0]
R2 = signal.convolve2d(R,M,boundary='symm',mode='same')

G = data[:,:,1]
G2 = signal.convolve2d(G,M,boundary='symm',mode='same')

B = data[:,:,2]
B2 = signal.convolve2d(B,M,boundary='symm',mode='same')
data2 = data.copy()
data2[:,:,0] = R2.astype('uint8')
data2[:,:,1] = G2.astype('uint8')
data2[:,:,2] = B2.astype('uint8')
I2 = Image.fromarray(data2,'RGB')
I2.show()


#---------------第一題------------------gaussian filter
"""
x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
d = np.sqrt(x*x+y*y)
sigma, mu = 1.0, 0.0
M = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
sumM = sum(sum(M))
M = M / sumM

R = data[:,:,0]
R2 = signal.convolve2d(R,M,boundary='symm',mode='same')

G = data[:,:,1]
G2 = signal.convolve2d(G,M,boundary='symm',mode='same')

B = data[:,:,2]
B2 = signal.convolve2d(B,M,boundary='symm',mode='same')
data2 = data.copy()
data2[:,:,0] = R2.astype('uint8')
data2[:,:,1] = G2.astype('uint8')
data2[:,:,2] = B2.astype('uint8')

I2 = Image.fromarray(data2,'RGB')
I2.show()
"""

#---------------第三題------------------sharpening filter
#銳化 小的標準差 - 大的標準差 窄的normal 減掉 寬的normal
"""
x, y = np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3))
d = np.sqrt(x*x+y*y)
sigma, mu = 0.99, 0.0
M1 = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
sumM1 = sum(sum(M1))
M1 = M1 / sumM1

x, y = np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3))
d = np.sqrt(x*x+y*y)
sigma, mu = 0.6, 0.0
M2 = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
sumM2 = sum(sum(M2))
M2 = M2 / sumM2
M = M2 - M1




#M = np.array([[-1,-1,-1],[-1,16,-1],[-1,-1,-1]]) / 8
M = np.array([[0,0,0],[0,2,0],[0,0,0]]) - np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9 
R = data[:,:,0]
R2 = signal.convolve2d(R,M,boundary='symm',mode='same')

G = data[:,:,1]
G2 = signal.convolve2d(G,M,boundary='symm',mode='same')

B = data[:,:,2]
B2 = signal.convolve2d(B,M,boundary='symm',mode='same')
data2 = data.copy()
data2[:,:,0] = R2.astype('uint8')
data2[:,:,1] = G2.astype('uint8')
data2[:,:,2] = B2.astype('uint8')


I2 = Image.fromarray(data2,'RGB')
I2.show()
"""
#只留邊界下來，長得像素描的圖
#先把原本的圖變成灰階，把RGB三個值相加起來，值最大的前20%標黑色 剩下標白色
#-------------------第四題------------------畫出edge圖


"""
I = I.convert("L") 
data = np.asarray(I)
data3 = np.zeros((640,640,3)).astype('uint8')
data4 = np.zeros((640,640,3)).astype('uint8')

#套用sobel_x 和 sobel_y
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])


R = data
R3 = signal.convolve2d(R,sobel_x,boundary='symm',mode='same')


R = data
R4 = signal.convolve2d(R,sobel_y,boundary='symm',mode='same')

R = R3 ** 2 + R4 ** 2

r = R.flatten()
idxr = np.argsort(-r) #descending with index

for i in range(81920):  #取前20％
    x = idxr[i] // 640
    y = idxr[i] % 640
    R[x][y] = 0

for j in range(81920,409600): #剩下的80％
    a = idxr[j] // 640
    b = idxr[j] % 640
    R[a][b] = 255

data4 = R.astype('uint8')

I4 = Image.fromarray(data4)
I4.show()
"""