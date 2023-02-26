import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import listdir;from os.path import isfile, join
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error as mse
from numba import jit

plt.rcParams["figure.figsize"] = (40,20)

'''1. transfer data format from format A to B'''
'''2. fulfill nan with the mean of surrounding values'''
'''3. generate sym.csv with all the dates merged in single csv'''

@jit(nopython=True)
def out_of_sample(results, pred_x, pred_y, real_y):
    pred_x = np.insert(pred_x, 0, 1, axis=1)
    pred_y = np.dot(pred_x, results)
    r_squared = 1 - np.sum((real_y - pred_y)**2) / np.sum((real_y - np.mean(real_y))**2)
    return r_squared

@jit(nopython=True)
def run_ols(X, Y, index, window_size):
    X = np.insert(X, 0, 1, axis=1)
    y = Y[index-window_size:index]
    results = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = np.dot(np.insert(X0[index,:], 0, 1), results)
    y_true = Y[index]
    return y_true, y_hat

'''on mac'''
path = "/Users/kang/Data/"
data_path = path + "out_jump/"
out_path = path + 'out_ols_numba/'

onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
already_done = sorted([f for f in listdir(out_path) if isfile(join(out_path, f))])

result_lst = []
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

    X0 = dflst.iloc[:,1:-1]
    X0 = X0[['date',"intrSn","qty","volBuyQty","volSellQty","volBuyNotional","nrTrades","is_jump"]]
    y0 = dflst.iloc[:,-1]

    window_size = 3900
    rst_lst = []
    for index in tqdm(range(window_size, X0.shape[0]), leave=False):
        X = X0.iloc[index-window_size:index,:].to_numpy()
        y_true, y_hat = run_ols(X, y0.to_numpy(), index, window_size)
        rst_lst.append([y_true, y_hat])

    rst = pd.DataFrame(rst_lst)
    rst.columns = ["yTrue","yPred"]
    result = pd.concat([dflst.iloc[window_size:,[0,1,2,3]].reset_index().drop('index',axis=1), rst],axis=1)
    result.to_csv(out_path + file)
