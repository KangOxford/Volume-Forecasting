import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
from numba import jit

import scipy

'''on linux'''
path = "/home/kanli/forth/"
data_path = path + "out_disjoint5/"
out_path = path + 'out_disjoint5_numba/'
# data_path = path + "out_overlap15/"
# out_path = path + 'out_ols_overlap15_numba/'
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
try:
    # already_done = []
    already_done = sorted([f for f in listdir(out_path) if isfile(join(out_path, f))])
except:
    import os;os.mkdir(out_path)
    already_done = sorted([f for f in listdir(out_path) if isfile(join(out_path, f))])


@jit(nopython=True)
def ols(X, y):
    X = np.column_stack((np.ones(X.shape[0]), X))
    XT_X_pinv = np.linalg.pinv(X.T @ X)
    beta = XT_X_pinv @ X.T @ y
    return beta

'''disjoint'''
lst5 = ['timeHMs',
 'timeHMe',
 'intrSn',
 'qty',
 'volSellQty',
 'ntn',
 'volBuyNotional',
 'volSellNotional',
 'nrTrades',
 'ntr',
 'volBuyNrTrades_lit',
 'volSellNrTrades_lit',
 'ol_lb5_qty',
 'ol_lb5_volSellQty',
 'ol_lb5_ntn',
 'ol_lb5_volBuyNotional',
 'ol_lb5_volSellNotional',
 'ol_lb5_nrTrades',
 'ol_lb5_ntr',
 'jump_value',
 'is_jump']




for j in tqdm(range(len(onlyfiles))): # on mac4
    # j = 0#$
    file = onlyfiles[j]
    # if file in already_done:
    #     continue
    print(f">>>> {j}th {file}")
    df = pd.read_pickle(data_path + file)

    X0 = df.iloc[:, 1:-1]
    X0 = X0[lst5]
    y0 = df.iloc[:, -1]
    X  = X0.to_numpy()
    y  = y0.to_numpy()

    window_size = 3900
    rst_lst = []
    for index in tqdm(range(window_size, X.shape[0]), leave=False):
        # index = window_size #$
        X1 = X[index - window_size:index,:]
        y1 = y[index-window_size:index]
        beta = ols(X1, y1)
        y_hat = np.dot(np.append(1, X[index, :]), beta)

        y_true = y[index]
        rst_lst.append([y_true, y_hat])

    rst = pd.DataFrame(rst_lst)
    rst.columns = ["yTrue","yPred"]

    result = pd.concat([df.iloc[window_size:,[0,1,2,3]].reset_index().drop('index',axis=1), rst],axis=1)
    result.to_pickle(out_path + file)

