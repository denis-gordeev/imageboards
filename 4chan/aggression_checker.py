import pandas as pd
import csv
f = open('messages-b.csv')
b = f.readlines()
f.close()
f = open('messages-pol.csv')
pol = f.readlines()
f.close()

f = open('aggression.csv', 'a+')
agg = f.readlines()
agg_check = [a.split('\t')[1] for a in agg[1:]]
#for i in range(1+(len(agg)/2), len(b)-1):
for i in range(1, len(b)-1):
    print (i)
    if not b[i].split('\t')[1] in agg_check:
        print b[i].split('\t')[0] + ' ' + b[i].split('\t')[1]
        agg = input ("Is there any aggresion? 1 or 0: ")
        f.write(b[i].strip()+'\t'+str(agg)+'\n')
    if not pol[i].split('\t')[1] in agg_check:
        print pol[i].split('\t')[0] + ' ' + pol[i].split('\t')[1]
        agg = input ("Is there any aggresion? 1 or 0: ")
        f.write(pol[i].strip()+'\t'+str(agg)+'\n')
