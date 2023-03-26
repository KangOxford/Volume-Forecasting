import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
from numba import jit
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Lasso

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


import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def lasso(X, y, alpha):
    X = np.column_stack((np.ones(X.shape[0]), X))
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)  # Increase max_iter
    lasso.fit(X, y)
    beta = lasso.coef_
    # if lasso.n_iter_ < lasso.max_iter:
    #     print(f"Lasso model converged after {lasso.n_iter_} iterations.")
    # else:
    #     print(f"Lasso model did not converge after {lasso.max_iter} iterations.")
    return beta


for j in tqdm(range(len(onlyfiles))):  # on mac4
    file = onlyfiles[j]
    # if file in already_done:
    #     print(f"++++ {j}th {file} is already done before")
    #     continue
    print(f">>>> {j}th {file}")
    dflst = pd.read_pickle(data_path + file)


    X0 = dflst.iloc[:, 5:-1].to_numpy()
    y0 = dflst.iloc[:, -1].to_numpy()
    scaler = StandardScaler()
    X0 = scaler.fit_transform(X0)

    window_size = 250
    rst_lst = []
    for index in tqdm(range(X0.shape[0] - 2 * window_size), leave=False):
        X = X0[window_size + index : index + 2 * window_size, :]
        y = y0[window_size + index : index + 2 * window_size]

        beta = lasso(X, y, alpha=1.0)

        y_hat = max(0, np.dot(np.append(1, X0[index + 2 * window_size, :]), beta))
        y_hat = min(y_hat, max(y0[window_size + index : index + 2 * window_size + 1]))
        y_true = y0[index + 2 * window_size]
        rst_lst.append([y_true, y_hat])

    rst = pd.DataFrame(rst_lst)
    rst.columns = ["yTrue", "yPred"]

    result = pd.concat(
        [
            dflst.iloc[window_size:, [0, 1, 2, 3, 4]]
            .iloc[window_size:, :]
            .reset_index()
            .drop("index", axis=1),
            rst,
        ],
        axis=1,
    )
    result.to_pickle(out_path + file)
