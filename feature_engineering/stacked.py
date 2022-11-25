# -*- coding: utf-8 -*-
import pandas as pd
from os import listdir; from os.path import isfile, join
mypath = "/Users/kang/Desktop/Volume-Forecasting/data_processing/data/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
filename_list = [f[:-4] for f in onlyfiles]
single_1_list = []
overlap1_5_list = []
overlap1_5_10_list = []
disjoint1_5_list = []
disjoint1_5_10_list = []
for i in range(len(onlyfiles)):
    name = filename_list[i]
    # print(name)
    # print(name[7:])
    # print("-----")
    file = pd.read_csv(mypath+onlyfiles[i])
    exec('date_'+ name + " = file")
    if name[7:] == "single_1":
        single_1_list.append()
