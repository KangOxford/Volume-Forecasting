from scipy import stats
ic = lambda x, f: stats.spearmanr(x, f)[0]



import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
from numba import jit
from sklearn.preprocessing import StandardScaler
import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    path = "/home/kanli/seventh/"
else:print("Unknown operating system")
data_path = path + "02_raw_component/"
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
if __name__=="__main__":
    df_list = []
    for item in onlyfiles:
        df = pd.read_pickle(data_path+item)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(df.iloc[:,4:-1])
        df.iloc[:,4:-1] = scaler.transform(df.iloc[:,4:-1])
        df_list.append(df)
    dff = pd.concat(df_list)
    igpd = iter(dff.groupby("date"))
    Sr_list =[]
    while True:
        try:
            date, new = next(igpd)
            sr_list = []
            newigpd = iter(new.groupby("timeHMs"))
            while True:
                try:
                    timeHMs,df = next(newigpd)
                    f = df.VO
                    dct = {}
                    for i in range(4,df.shape[1]):
                    # for i in range(5,df.shape[1]-2):
                        x = df.iloc[:,i]
                        name = df.columns[i]
                        dct[name] = ic(x,f)
                        # dct[name] = ic(f,x)
                    sr = pd.Series(data=dct,name=timeHMs)
                    sr_list.append(sr)
                    print(timeHMs)
                except: break
            srdf = pd.concat(sr_list,axis=1)
            srdf['mean'] = srdf.apply(np.mean,axis=1)
            Sr = srdf['mean']
            Sr.name = date
            Sr_list.append(Sr)
        except:break
    Srdf = pd.concat(Sr_list, axis=1)
    Srdf['mean'] = Srdf.apply(np.mean, axis=1)
    Srdf = Srdf.sort_values("mean")

