# -*- coding: utf-8 -*-
import pandas as pd
from os import listdir; from os.path import isfile, join
path = "/Users/kang/Desktop/Volume-Forecasting/data_processing/data/"
subpath = 'disjoint1_5'
mypath = path + subpath +'/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = sorted(onlyfiles)    
filename_list = [f[:-4] for f in onlyfiles]
my_list = []
for i in range(len(onlyfiles)):
    name = filename_list[i]
    file = pd.read_csv(mypath+onlyfiles[i])
    exec('date_'+ name + " = file")
    exec('my_list.append('+"date_"+name+")")
for i in range(len(my_list)):
    df = my_list[i]
    df.symbol = int(df.symbol[0][:-1])
new = pd.concat(my_list, axis = 0)
gn = new.groupby('timeHM_start')
# df = next(gn)
# new = df.merge(right = my_list[1], on = 'timeHM_start', axis = 1)
    
for item in gn:
    df = item[1]
    break
for item in gn:
    target = item[1]
    df = pd.concat([df,target])
new_df = df.iloc[5:,:]
    
df1 = new_df.groupby('timeHM_start').mean().iloc[:,2:]
