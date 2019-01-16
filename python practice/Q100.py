# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""def cycle_length(x):
    
    if x == 1:
        return 1
    elif x % 2 == 1:
        return 1 + cycle_length(3 * x + 1)
    else:
        return 1 + cycle_length(x / 2)"""
        
while True:
    try:
        i, j =input().split()
        i = int(i) 
        j = int(j)

        while (i > j):
            temp = i
            i = j
            j = temp

        kk = [0] * 1000000
        maxmama = 0
        for n in range(i,j+1):
            number = n
            #print(n)
            cycle_length = 1
            while(n != 1):
                if kk[number] != 0 and kk[number] < 1000000:  # 如果kk[n]之前已經算過的話
                    if kk[number] > maxmama:
                        maxmama = kk[number] 
                    break
                else:
                    if n % 2 == 1:
                        n = 3 * n + 1
                    else:
                        n = n // 2
                cycle_length = cycle_length + 1 
            #print(cycle_length)
            kk[n] = cycle_length
            #print(kk[n])
            #print(kk)
            if kk[n] > maxmama:
                maxmama = kk[n] 
                       
        print(i,j,maxmama)
    except EOFError:
        break

