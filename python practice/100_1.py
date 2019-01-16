# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:54:00 2017

@author: user
"""

#a=int(input())#輸入第一個值
#b=int(input())#輸入第二個值

a,b=(int(x) for x in input().split())
ra=a
rb=b
maxcount=0
count=[0] * 1000000 # 暫存list

while(a,b<1000001):
    
    while(a>b):
        c=a
        a=b
        b=c

    for n in range(a,b+1):
        i = n        
        length = 1 #cycle_length的長度(因為會算本身)
        while n!=1 and n < 1000001:
            if count[n] != 0:
                length += count[n]
                length -= 1
                break
            else:
                if n%2==1:
                    n=3*n+1
                else:
                    n=n//2
            length = length + 1
        count[i] = length  #例如：count[10] = 20
        #print(count[i])
            #count.append(n)
        """if (len(count)>maxcount): 好像這樣寫會輸出count list的長度而不是每個數的總共算了多少次
            maxcount=len(count)"""  
        if count[i] > maxcount:
            maxcount = count[i] 
    print(ra,rb,maxcount)
    
    a,b=(int(x) for x in input().split())
    ra=a
    rb=b
    maxcount=0
    count=[0] * 1000000