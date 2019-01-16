#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 00:06:03 2017

@author: Rong
"""



for num in range(0,int(input())):
    result = ""
    count = 0    
    s = input() 
    c = s[0]           
    print ("Case %d: " % (num+1),end='')
    for i in range(0,len(s)):        
        if s[i] >= '0' and s[i] <= '9':
            count = count * 10 + int(s[i]) - int('0')
        else:  # if s[i] >= 'A' and s[i] <= 'Z':
            for j in range(0,count):
                result += c
            count = 0
            c = s[i]          
    for k in range(0,count):        
        result += c    
    print(result,end='')
    print()