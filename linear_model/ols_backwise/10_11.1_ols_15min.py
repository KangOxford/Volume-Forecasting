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

@jit(nopython=True)
def ols(X, y):
    X = np.column_stack((np.ones(X.shape[0]), X))
    XT_X_pinv = np.linalg.pinv(X.T @ X)
    beta = XT_X_pinv @ X.T @ y
    return beta




for i in tqdm(range(len(onlyfiles))): # on mac4
    bin_size = 26
    num_day = 10
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

    r2_score_list = []
    times = np.arange(0,dflst.shape[0]//bin_size-num_day)
    for time in times:
        index = bin_size * time
        X_train = dflst.iloc[index:index+bin_size*num_day,4:-1]
        y_train = dflst.iloc[index:index+bin_size*num_day:,-1]
        X_test = dflst.iloc[index+bin_size*num_day:index+bin_size*(num_day+1),4:-1]
        y_test = dflst.iloc[index+bin_size*num_day:index+bin_size*(num_day+1),-1].values
        from sklearn.linear_model import Lasso
        reg = Lasso(alpha=1)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        y = pd.DataFrame([y_test,y_pred]).T
        r2_score = reg.score(X_test,y_test)
        date = dflst.date.iloc[index+bin_size*num_day]

        r2_score_list.append([date,r2_score])


        # print('R squared training set', round(reg.score(X_train, y_train) * 100, 2))
        # from sklearn.metrics import mean_squared_error
        # # Training data
        # pred_train = reg.predict(X_train)
        # mse_train = mean_squared_error(y_train, pred_train)
        # print('MSE training set', round(mse_train, 2))

    r2_score_arr = np.array(r2_score_list)

    fig_path = path + "04_pred_true_fig/"
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # x_axis = np.arange(len(values))
    dates = r2_score_arr[:,0]
    values = r2_score_arr[:,1].astype(np.float32)
    dates = pd.to_datetime(dates, format='%Y%m%d')
    x_axis = dates
    plt.plot(x_axis, values)
    plt.plot(x_axis, np.full(len(values), values.mean()), label=f'Mean: {values.mean():.2f}')  # mean value
    # Set the x-axis format to display dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    # Rotate the x-axis tick labels to avoid overlap
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date')
    plt.ylabel('R2 Score')
    title = "R2 Score for Single Stock " + file[:-4]
    plt.title(title)
    plt.legend()
    plt.savefig(fig_path+title)
    plt.show()