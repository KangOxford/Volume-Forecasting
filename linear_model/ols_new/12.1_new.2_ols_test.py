import pandas as pd
import numpy as np
import os;from os import listdir;from os.path import isfile, join
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt; plt.rcParams["figure.figsize"] = (12, 8)
import matplotlib.dates as mdates
from sklearn.metrics import r2_score
import inspect
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt; plt.rcParams["figure.figsize"] = (12, 8)
import matplotlib.dates as mdates
from sklearn.metrics import r2_score


'''transform'''
import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    path = "/home/kanli/seventh/"
else:print("Unknown operating system")
data_path = path + "05_result_data_path/"

# try: listdir(out_path)
# except:import os;os.mkdir(out_path)

y_df = pd.read_csv(data_path+"y_true_pred.csv",index_col=0)

bin_size = 26
y_df['intra_interval'] = np.tile(np.array([*[0]*2,*[1]*22,*[2]*2]), y_df.shape[0]//bin_size)


igroupped = iter(y_df.groupby("intra_interval"))
open = next(igroupped)[1]
mid = next(igroupped)[1]
close = next(igroupped)[1]

# ==========================


# interval = [open,close][0]
interval = [open,close][1]
groupped_interval = interval.groupby("symbol")
info_arr_list = []
for symbol, item in groupped_interval:
    info_list = []
    for i in range(0,item.shape[0]//10):
        slice = item.iloc[10*i:10*(i+1),[2,3]]
        date = item.date.iloc[10*i]
        r2_value = r2_score(slice.true,slice.pred)
        mse_value = mean_squared_error(slice.true,slice.pred)
        info = np.array([symbol, date.astype(str),r2_value.astype(np.float32),mse_value.astype(np.float32)])
        info_list.append(info)
    info_arr = np.array(info_list)
    info_arr_list.append(info_arr)
info_arr_arr = np.array(info_arr_list).reshape(-1,4)


r2_df = pd.DataFrame(info_arr_arr[:,[0,1,2]],columns=['symbol','date','r2'])
mse_df = pd.DataFrame(info_arr_arr[:,[0,1,3]],columns=['symbol','date','mse'])

def plot(r2_df,name):
    r2_df.symbol = r2_df.symbol.astype(str)
    r2_df.date = r2_df.date.astype(np.int64)
    r2_df.iloc[:,-1] = r2_df.iloc[:,-1].astype(np.float32)
    mean = r2_df.groupby("date").mean()
    std = r2_df.groupby("date").std()
    mean_add_std = mean + std
    mean_minus_std = mean - std
    mean_std = pd.concat([mean_minus_std,mean,mean_add_std],axis=1)


    fig_path = path + "06_analysis_fig2/"
    try: listdir(fig_path)
    except:import os;os.mkdir(fig_path)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # x_axis = np.arange(len(values))
    dates = mean_std.index
    values0 = mean_std.iloc[:, 0].astype(np.float32)
    values1 = mean_std.iloc[:, 1].astype(np.float32)
    values2 = mean_std.iloc[:, 2].astype(np.float32)
    dates = pd.to_datetime(dates, format='%Y%m%d')
    x_axis = dates
    plt.plot(x_axis, values0, label=f'Mean-Std: {values0.mean():.2f}')
    plt.plot(x_axis, values1, label=f'Mean: {values1.mean():.2f}')
    plt.plot(x_axis, values2, label=f'Mean+Std: {values2.mean():.2f}')
    # plt.plot(x_axis, np.full(len(values), values.mean()), label=f'Mean: {values.mean():.2f}')  # mean value
    # Set the x-axis format to display dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    # Rotate the x-axis tick labels to avoid overlap
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date')
    plt.ylabel(name + ' Score')
    # title = name + " Score for Single Stock " + file[:-4]
    # plt.title(title)
    plt.legend()
    # plt.savefig(fig_path + title)
    plt.show()
plot(r2_df,name = "R2")
# plot(mse_df,name = "MSE")

# ==========================


interval = mid
groupped_interval = interval.groupby("symbol")
info_arr_list = []
for symbol, item in groupped_interval:
    info_list = []
    for i in range(0,item.shape[0]//110):
        slice = item.iloc[110*i:110*(i+1),[2,3]]
        date = item.date.iloc[110*i]
        r2_value = r2_score(slice.true,slice.pred)
        mse_value = mean_squared_error(slice.true,slice.pred)
        info = np.array([symbol, date.astype(str),r2_value.astype(np.float32),mse_value.astype(np.float32)])
        info_list.append(info)
    info_arr = np.array(info_list)
    info_arr_list.append(info_arr)
info_arr_arr = np.array(info_arr_list).reshape(-1,4)


r2_df = pd.DataFrame(info_arr_arr[:,[0,1,2]],columns=['symbol','date','r2'])
mse_df = pd.DataFrame(info_arr_arr[:,[0,1,3]],columns=['symbol','date','mse'])

def plot(r2_df,name):
    r2_df.symbol = r2_df.symbol.astype(str)
    r2_df.date = r2_df.date.astype(np.int64)
    r2_df.iloc[:,-1] = r2_df.iloc[:,-1].astype(np.float32)
    mean = r2_df.groupby("date").mean()
    std = r2_df.groupby("date").std()
    mean_add_std = mean + std
    mean_minus_std = mean - std
    mean_std = pd.concat([mean_minus_std,mean,mean_add_std],axis=1)


    fig_path = path + "06_analysis_fig2/"
    try: listdir(fig_path)
    except:import os;os.mkdir(fig_path)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # x_axis = np.arange(len(values))
    dates = mean_std.index
    values0 = mean_std.iloc[:, 0].astype(np.float32)
    values1 = mean_std.iloc[:, 1].astype(np.float32)
    values2 = mean_std.iloc[:, 2].astype(np.float32)
    dates = pd.to_datetime(dates, format='%Y%m%d')
    x_axis = dates
    plt.plot(x_axis, values0, label=f'Mean-Std: {values0.mean():.2f}')
    plt.plot(x_axis, values1, label=f'Mean: {values1.mean():.2f}')
    plt.plot(x_axis, values2, label=f'Mean+Std: {values2.mean():.2f}')
    # plt.plot(x_axis, np.full(len(values), values.mean()), label=f'Mean: {values.mean():.2f}')  # mean value
    # Set the x-axis format to display dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    # Rotate the x-axis tick labels to avoid overlap
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date')
    plt.ylabel(name + ' Score')
    # title = name + " Score for Single Stock " + file[:-4]
    # plt.title(title)
    plt.legend()
    # plt.savefig(fig_path + title)
    plt.show()
plot(r2_df,name = "R2")
plot(mse_df,name = "MSE")
