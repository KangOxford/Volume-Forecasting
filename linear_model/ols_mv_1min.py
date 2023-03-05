import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
from numba import jit


import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
    data_path = path + "raw_component/"
    out_path = path + "ols_mv_1min/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    # path = "/home/kanli/forth/"
    # data_path = path + "out_jump/"
    # out_path = path + "out_disjoint5/"
else:print("Unknown operating system")



onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
already_done = sorted([f for f in listdir(out_path) if isfile(join(out_path, f))])

@jit(nopython=True)
def ols(X, y):
    X = np.column_stack((np.ones(X.shape[0]), X))
    XT_X_pinv = np.linalg.pinv(X.T @ X)
    beta = XT_X_pinv @ X.T @ y
    return beta

for j in tqdm(range(399,299,-1)): # on mac4
    file = onlyfiles[j]
    if file in already_done:
        continue
    print(f">>>> {j}th {file}")
    df = pd.read_csv(data_path + file, index_col=0)

    df_list = []
    gpd = df.groupby("date")
    for index, item in gpd:
        item['VO'] = item.qty.shift(-1)
        item = item.dropna()
        df_list.append(item)
    dflst = pd.DataFrame(pd.concat(df_list))

    X0 = dflst.iloc[:,1:-1].to_numpy()
    X0 = X0[:, [0, 1, 2, 3, 4, 5, 6, -1]]
    y0 = dflst.iloc[:,-1].to_numpy()

    window_size = 3900
    rst_lst = []
    for index in tqdm(range(window_size, X0.shape[0]), leave=False):
        X = X0[index-window_size:index,:]
        y = y0[index-window_size:index]
        beta = ols(X, y)
        y_hat = np.dot(np.append(1, X0[index,:]), beta)
        y_true = y0[index]
        rst_lst.append([y_true, y_hat])

    rst = pd.DataFrame(rst_lst)
    rst.columns = ["yTrue","yPred"]

    result = pd.concat([dflst.iloc[window_size:,[0,1,2,3]].reset_index().drop('index',axis=1), rst],axis=1)
    result.to_csv(out_path + file)
