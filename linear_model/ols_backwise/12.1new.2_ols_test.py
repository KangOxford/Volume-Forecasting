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
    data_path = path + "05_result_data_path/"
    # out_path = path + "04_pred_true_fig/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    path = "/home/kanli/fifth/"
    data_path = path + "05_result_data_path/"
    # out_path = path + "05_result_data_path/"
else:print("Unknown operating system")

# try: listdir(out_path)
# except:import os;os.mkdir(out_path)

r2_df = pd.read_csv(data_path+"r2_score.csv",index_col=0)
mse_df = pd.read_csv(data_path+"mse_score.csv",index_col=0)
y_df = pd.read_csv(data_path+"y_true_pred.csv",index_col=0)

bin_size = 26
y_df['intra_interval'] = np.tile(np.array([*[0]*2,*[1]*22,*[2]*2]), y_df.shape[0]//bin_size)


igroupped = iter(y_df.groupby("intra_interval"))
open = next(igroupped)[1]
mid = next(igroupped)[1]
close = next(igroupped)[1]

interval = [open,close][0]

info_list = []
for i in range(0,interval.shape[0]//10):
    slice = interval.iloc[10*i:10*(i+1),[2,3]]
    date = interval.date.iloc[10*i]
    r2_value = r2_score(slice.true,slice.pred)
    mse_value = mean_squared_error(slice.true,slice.pred)
    info = np.array([date.astype(str),r2_value.astype(np.float32),mse_value.astype(np.float32)])
    info_list.append(info)
info_arr = np.array(info_list)
# info_df = pd.DataFrame(info_arr,columns=['date','r2','mse'])
r2_df = pd.DataFrame(info_arr[:,[0,1]],columns=['date','r2'))
mse_df = pd.DataFrame(info_arr[:,[0,2]],columns=['date','mse')

def plot(r2_df,name):
    mean = r2_df.groupby("date").mean()
    std = r2_df.groupby("date").std()
    mean_add_std = mean + std
    mean_minus_std = mean - std
    mean_std = pd.concat([mean_minus_std,mean,mean_add_std],axis=1)


    fig_path = path + "06_analysis_fig/"
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
