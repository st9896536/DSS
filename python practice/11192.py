# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

    

x = input()
tmp = x.split()
num = int(tmp[0])
while(num!=0):
    str = tmp[1]
    count = len(str) // num
    for i in range(0,len(str),count):
        for j in range(count-1,-1,-1):
            print(str[i+j],end='')
    print('')
    x = input()
    tmp = x.split()
    num = int(tmp[0])