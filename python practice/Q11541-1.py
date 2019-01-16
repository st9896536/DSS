#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 00:06:03 2017

@author: Rong
"""



for num in range(0,int(input())):
    result = [] # 暫存的list
    ans = [] #最後印出
    ss = ""
    s = input()
    count = 0       
    print ("Case %d: " % (num+1),end='')
    for i in range(0,len(s)):        
        if s[i] >= 'A' and s[i] <= 'Z':
            result.append(s[i])            
        elif s[i] >= '0' and s[i] <= '9':
            if result[-1] >= '0' and result[-1] <= '9':
                result[-1] = result[-1] + s[i]
            else:
                result.append(s[i])
    for i in range(0,len(result),2):
        result[i+1] = int(result[i+1])
        #print(result[i+1])
        ans.append(result[i] * result[i+1])
    ss = ''.join(map(str,ans) )
    print(ss)
    #print(result)
    