import pandas as pd
import numpy as np
import os;from os import listdir;from os.path import isfile, join
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt; plt.rcParams["figure.figsize"] = (12, 8)
import matplotlib.dates as mdates
from sklearn.metrics import r2_score
import inspect


# regulator = "OLS"
regulator = "Lasso"
# regulator = "Ridge"


'''transform'''
import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    path = "/home/kanli/seventh/"
else:print("Unknown operating system")
data_path = path + "05_result_data_path/"+regulator+"/"

# try: listdir(out_path)
# except:import os;os.mkdir(out_path)
r2_df = pd.read_csv(data_path+"r2_score.csv",index_col=0)
mse_df = pd.read_csv(data_path+"mse_score.csv",index_col=0)
y_df = pd.read_csv(data_path+"y_true_pred.csv",index_col=0)


def plot(r2_df,name):
    mean = r2_df.groupby("date").mean()
    std = r2_df.groupby("date").std()
    mean_add_std = mean + std
    mean_minus_std = mean - std
    mean_std = pd.concat([mean_minus_std,mean,mean_add_std],axis=1)


    try: listdir(path + "06_analysis_fig/")
    except:import os;os.mkdir(path + "06_analysis_fig/")
    fig_path = path + "06_analysis_fig/" + regulator+"/"
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
    title = name + " Score for Averaged"
    plt.title(title)
    plt.legend()
    plt.savefig(fig_path + title)
    plt.show()

def filter_date(r2_df):
    option_exprire_list= [20170721,20170818,20170915,20171020,20171117,20171215]
    strange_list = [] # default
    # strange_list = [20171019,20171114,20171129] # Lasso
    # strange_list = [20170912, 20171019] # Ridge
    # strange_list = [20170811,20170720,20170815,20171019,20171027,20170725,20170803,20170802] #ols
    # # filter_date_r2_df.date.iloc[filter_date_r2_df.value.argmin()]
    outstanding_date_list = option_exprire_list + strange_list
    re = r2_df[~r2_df['date'].isin(outstanding_date_list)]
    return re
filter_date_r2_df = filter_date(r2_df)
filter_date_mse_df = filter_date(mse_df)
plot(filter_date(mse_df),name = "MSE")
plot(filter_date_r2_df,name = "R2")


