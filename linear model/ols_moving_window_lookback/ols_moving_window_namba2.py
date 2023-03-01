import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
from numba import jit

import scipy

'''on linux'''
path = "/home/kanli/forth/"
data_path = path + "out_overlap15/"
out_path = path + 'out_ols_overlap15_numba/'
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

def find_significant_cols(X, y, alpha=0.05):
    beta = ols(X, y)
    n, k = X.shape
    dof = n - k - 1
    t_crit = scipy.stats.t.ppf(1 - alpha/2, dof)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)))  # standard errors
    t_vals = beta / se  # t-statistics
    p_vals = scipy.stats.t.sf(np.abs(t_vals), dof) * 2  # two-sided p-values
    significant_cols = np.where(p_vals <= alpha)[0]  # significant columns
    return significant_cols

def stepwise_regression(X, y, alpha=0.05):
    k = X.shape[1]
    significant_cols = []
    while len(significant_cols) < k:
        remaining_cols = np.setdiff1d(range(k), significant_cols)
        X_subset = X[:, remaining_cols]
        significant_cols_subset = find_significant_cols(X_subset, y, alpha)
        significant_cols_subset = remaining_cols[significant_cols_subset]
        if len(significant_cols_subset) == 0:
            break
        significant_cols = np.union1d(significant_cols, significant_cols_subset)
    return significant_cols


for j in tqdm(range(len(onlyfiles))): # on mac4
    j = 0#$
    file = onlyfiles[j]
    # if file in already_done:
    #     continue
    print(f">>>> {j}th {file}")
    df = pd.read_pickle(data_path + file)

    X0 = df.iloc[:, 1:-1]
    y0 = df.iloc[:, -1]
    X  = X0.to_numpy()
    y  = y0.to_numpy()

    window_size = 3900
    rst_lst = []
    for index in tqdm(range(window_size, X.shape[0]), leave=False):
        index = window_size #$
        X = X[index - window_size:index,:]
        y = y[index-window_size:index]
        beta, p_values = ols(X, y)
        y_hat = np.dot(np.append(1, X[index,:]), beta)
        y_true = y[index]
        rst_lst.append([y_true, y_hat])

    rst = pd.DataFrame(rst_lst)
    rst.columns = ["yTrue","yPred"]


    # result = pd.concat([df.iloc[window_size:,[0,1,2,3]].reset_index().drop('index',axis=1), rst],axis=1)
    # result.to_pickle(out_path + file)

