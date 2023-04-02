import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
from numba import jit
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Normalizer


import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
    data_path = path + "02_raw_component/"
    out_path = path + '03_out_15min_pred_true_pairs_after_ols/'
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    path = "/home/kanli/fifth/"
    data_path = path + "raw_component15/"
    out_path = path + 'out_15min_pred_true_pairs_after_ols/'
else:print("Unknown operating system")

try: listdir(out_path)
except:import os;os.mkdir(out_path)

onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
already_done = sorted([f for f in listdir(out_path) if isfile(join(out_path, f))])
cols = pd.read_csv(path + "ols_feat_15min.csv",index_col=0).index.to_list()[1:]

@jit(nopython=True)
def ols(X, y):
    X = np.column_stack((np.ones(X.shape[0]), X))
    XT_X_pinv = np.linalg.pinv(X.T @ X)
    beta = XT_X_pinv @ X.T @ y
    return beta

for j in tqdm(range(len(onlyfiles))): # on mac4
    file = onlyfiles[j]
    # if file in already_done:
    #     print(f"++++ {j}th {file} is already done before")
    #     continue
    print(f">>>> {j}th {file}")
    dflst = pd.read_pickle(data_path + file)[['symbol', 'date', 'timeHMs', 'timeHMe', 'intrSn']+cols+['VO']]
    dflst.groupby(["date"]).apply(np.count)


    X0 = dflst.iloc[:,5:-1].to_numpy()
    y0 = dflst.iloc[:,-1].to_numpy()
    # col norm of x
    scaler = StandardScaler()
    X0 = scaler.fit_transform(X0)

    window_size = 260
    rst_lst = []
    for index in tqdm(range(X0.shape[0]-2*window_size), leave=False):
        X = X0[window_size+index:index+2*window_size,:]
        y = y0[window_size+index:index+2*window_size]

        beta = ols(X, y)
        min_limit = y.min()
        max_limit = y.max()
        y_hat = max(min_limit, np.dot(np.append(1, X0[index+2*window_size,:]), beta)) # add bottom
        y_hat = min(y_hat, max_limit) # add ceiling
        # y_hat = np.dot(np.append(1, X0[index+2*window_size,:]), beta)
        y_true = y0[index+2*window_size]
        rst_lst.append([y_true, y_hat])

    rst = pd.DataFrame(rst_lst)
    rst.columns = ["yTrue","yPred"]

    result = pd.concat([dflst.iloc[window_size:,[0,1,2,3,4]].iloc[window_size:,:].reset_index().drop('index',axis=1), rst],axis=1)
    result.to_pickle(out_path + file)
