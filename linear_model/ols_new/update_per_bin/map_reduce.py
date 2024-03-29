import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
from numba import jit
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Normalizer
from linear_model.ols_new.update_per_bin.regularity import regularity_ols

# import warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings('ignore', category=ConvergenceWarning)


import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    path = "/home/kanli/seventh/"
else:print("Unknown operating system")
data_path = path + "02_raw_component/"
out_path = path + '03_out_15min_pred_true_pairs_after_ols/'

try: listdir(out_path)
except:import os;os.mkdir(out_path)


onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
already_done = sorted([f for f in listdir(out_path) if isfile(join(out_path, f))])

@jit(nopython=True)
def ols(X, y):
    X = np.column_stack((np.ones(X.shape[0]), X))
    XT_X_pinv = np.linalg.pinv(X.T @ X)
    beta = XT_X_pinv @ X.T @ y
    return beta


bin_size = 26
num_day = 10

import mr4mp

def trial(time,dflst):
    index = bin_size * time
    y_list = []
    for drift in range(bin_size):
        X_train = dflst.iloc[index + drift:index + bin_size * num_day + drift, 4:-1]
        y_train = dflst.iloc[index + drift:index + bin_size * num_day + drift, -1]
        X_test = dflst.iloc[index + bin_size * num_day + drift, 4:-1]
        y_test = dflst.iloc[index + bin_size * num_day + drift, -1]
        y_pred = regularity_ols(X_train, y_train, X_test, regulator)
        min_limit, max_limit = y_train.min(), y_train.max()
        y_pred = min(max(min_limit, y_pred), max_limit)
        y_list.append([y_test, y_pred])
    y_arr = np.array(y_list)
    y_test = y_arr[:, 0];
    y_pred = y_arr[:, 1]
    from sklearn.metrics import r2_score
    r2_score_value = r2_score(y_test, y_pred)
    from sklearn.metrics import mean_squared_error
    mse_score_value = mean_squared_error(y_test, y_pred)
    date = dflst.date.iloc[index + bin_size * num_day]
    y_true_pred = np.array(
        [np.full(bin_size, file[:-4]).astype(str), np.full(bin_size, date).astype(str), y_test,
         y_pred.astype(np.float32)]).T
    return [file[:-4], date, r2_score_value, mse_score_value, y_true_pred]


def combine(results,dflst):
    r2_score_list = []
    mse_score_list = []
    y_true_pred_list = []
    for res in results:
        r2_score_list.append([res[0], res[1], res[2]])
        mse_score_list.append([res[0], res[1], res[3]])
        y_true_pred_list.append(res[4])
    return [r2_score_list, mse_score_list, y_true_pred_list]


array1 = np.concatenate( [np.arange(1,10,0.01), np.arange(10,50,0.1) ])
array2 = np.arange(1,0.001,-0.001)
combined_array = np.array(list(zip(array1, array2))).flatten()
# used for alphas


# regulator = "Ridge"
regulator = "Lasso"
# regulator = "OLS"

if __name__=="__main__":
    r2_score_arr_list = []
    mse_score_arr_list = []
    y_true_pred_arr_list = []
    for i in tqdm(range(len(onlyfiles)),leave=True): # on mac4

        file = onlyfiles[i]
        # if file in already_done:
        #     print(f"++++ {j}th {file} is already done before")
        #     continue
        print(f">>>> {i}th {file}")
        dflst = pd.read_pickle(data_path + file)
        counted = dflst.groupby("date").count()
        date = pd.DataFrame(counted[counted['VO'] ==26].index)
        dflst = pd.merge(dflst, date, on = "date")
        assert dflst.shape[0]/ bin_size ==  dflst.shape[0]// bin_size

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(dflst.iloc[:,4:-1])
        dflst.iloc[:,4:-1] = scaler.transform(dflst.iloc[:,4:-1])

        # mapreduce {


        pool = mr4mp.pool()
        # results = pool.mapreduce(trial, combine, np.arange(0, dflst.shape[0] // bin_size - num_day))
        results = pool.mapreduce(trial, combine, np.arange(0, dflst.shape[0] // bin_size - num_day),
                                 [dflst] * np.arange(0, dflst.shape[0] // bin_size - num_day))
        r2_score_list, mse_score_list, y_true_pred_list = results

        # mapreduce }

    r2_score_arr_arr = np.array(r2_score_arr_list).reshape(-1,3)
    mse_score_arr_arr = np.array(mse_score_arr_list).reshape(-1,3)
    y_true_pred_arr_arr = np.array(y_true_pred_arr_list).reshape(-1,4)

    r2_score_arr_df = pd.DataFrame(r2_score_arr_arr,columns=["symbol",'date','value'])
    mse_score_arr_df = pd.DataFrame(mse_score_arr_arr,columns=["symbol",'date','value'])
    y_true_pred_arr_df = pd.DataFrame(y_true_pred_arr_arr,columns=["symbol",'date','true','pred'])


    try:listdir(path + "05_result_data_path/")
    except:import os;os.mkdir(path + "05_result_data_path/")

    result_data_path = path + "05_result_data_path/"+regulator+"/"
    try:listdir(result_data_path)
    except:import os;os.mkdir(result_data_path)

    r2_score_arr_df.to_csv(result_data_path + "r2_score.csv")
    mse_score_arr_df.to_csv(result_data_path + "mse_score.csv")
    y_true_pred_arr_df.to_csv(result_data_path + "y_true_pred.csv")



    # array1 = np.concatenate( [np.arange(1,10,0.01), np.arange(10,50,0.1) ])
    # array2 = np.arange(1,0.001,-0.001)
    # combined_array = np.array(list(zip(array1, array2))).flatten()
    # print(combined_array)
