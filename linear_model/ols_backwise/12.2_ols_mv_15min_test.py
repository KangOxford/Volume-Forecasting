import pandas as pd
import os;from os import listdir;from os.path import isfile, join
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt; plt.rcParams["figure.figsize"] = (12, 8)
import pandas as pd
import numpy as np
import os;from os import listdir;from os.path import isfile, join
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt; plt.rcParams["figure.figsize"] = (12, 8)
import matplotlib.dates as mdates
from sklearn.metrics import r2_score

'''transform'''
import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
    data_path = path + "03_out_15min_pred_true_pairs_after_ols_intraSession/"
    out_path = path + "04_pred_true_fig_intraSession/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    path = "/home/kanli/fifth/"
    data_path = path + "out_15min_pred_true_pairs_after_ols/"
    out_path = path + "04_pred_true_fig/"
else:print("Unknown operating system")

try: listdir(out_path)
except:import os;os.mkdir(out_path)

old_data_path = data_path
old_out_path = out_path
r_squared_list = []
for jump in ['open/','mid/','close/']:
    # jump = 'open/'
    # jump = 'mid/'
    jump = 'close/'
    data_path = old_data_path + jump
    out_path = old_out_path + jump
    try:
        listdir(out_path)
    except:
        import os;os.mkdir(out_path)

    onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
    df_lst = pd.concat(map(lambda file: pd.read_pickle(data_path + file), onlyfiles))

    df_lst = df_lst[df_lst.date != '20170801']
    df_lst = df_lst[df_lst.date != '20171229']
    # groupby symbol
    r2_scores_list = []
    igpd = iter(df_lst.groupby('symbol'))
    window_size = 26*20
    # window_size = 2*20
    # window_size = 2*50
    item_list = []

    while True:
        try:
            symbol, item = next(igpd)
            r2_list = []
            for index in range(item.shape[0]-window_size):
                new = item.iloc[index:index+window_size, :]
                r2 = r2_score(new.yTrue, new.yPred)
                # r2 = mean_squared_error(new.yTrue, new.yPred)
                r2_list.append(r2)
            assert len(r2_list) + window_size == item.shape[0]
            item['r2'] = pd.Series(r2_list)
            item.dropna(inplace = True)
            item_list.append(item)
            print(symbol, item.r2.mean())
        except StopIteration: break
    item_df = pd.concat(item_list)
    mean_date = pd.DataFrame(item_df.groupby('date').mean()['r2'])

    value = mean_date
    value['date'] = pd.to_datetime(value.index, format='%Y%m%d')
    plt.plot(value.date, value.r2, label = 'r2')
    # plt.plot(value.date, value.r2, label = 'mse')
    # Set the x-axis format to display dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    # Rotate the x-axis tick labels to avoid overlap
    plt.gcf().autofmt_xdate()
    title = "yPred_&_yTrue ols15min rolling_r2(20_days)_by_date_open_interval"
    # title = "yPred_&_yTrue ols15min rolling_mse(20_days)_by_date_open_interval"
    plt.title(title)
    plt.legend();
    plt.savefig(out_path + title + '.png')
    plt.show()














    # groupby date
    mean_date = df_lst.groupby('date').mean()
    r_squared_mean_date = r2_score(mean_date['yTrue'], mean_date['yPred'])
    mse_date = df_lst.groupby('date').apply(lambda x: mean_squared_error(x['yTrue'],x['yPred']))
    # groupby date and time
    df_lst['date_time'] = df_lst.date.apply(str) + df_lst.timeHMs.apply(lambda x: str(x).zfill(4))
    mean_date_time = df_lst.groupby('date_time').mean()
    r_squared_mean_date_time = r2_score(mean_date_time['yTrue'], mean_date_time['yPred'])
    mse_date_time = df_lst.groupby('date_time').apply(lambda x: mean_squared_error(x['yTrue'],x['yPred']))
    r_squared_list.append({jump:[r_squared_mean_date, r_squared_mean_date_time]})



    def plot_pred_true(value,title, size=(12, 8)):
        plt.rcParams["figure.figsize"] = size
        plt.plot(value.yTrue.values, label='True')
        plt.plot(value.yPred.values, label='Pred')
        plt.title(title)
        plt.legend();
        fig_name = out_path + title +'.png'
        if os.path.exists(fig_name): os.remove(fig_name)
        plt.savefig(fig_name)
        plt.show()

    def plot_mse(value, title, size=(12, 8)):
        plt.rcParams["figure.figsize"] = size
        plt.plot(value.values, label="MSE");
        plt.title(title)
        plt.legend();
        plt.savefig(out_path + title + '.png')
        plt.show()

    '''plot'''
    '''date'''
    plot_pred_true(mean_date,"yPred_&_yTrue ols15min mean_by_date")
    plot_mse(mse_date,"yPred_&_yTrue ols15min mse_by_date")

    '''date_time'''
    plot_pred_true(mean_date_time,"yPred_&_yTrue ols15min mean_by_date_&_time",(40,10))
    plot_mse(mse_date_time,"yPred_&_yTrue ols15min mse_by_date_&_time",(40,10))



    # '''anomaly'''
    # '''anomaly handling'''
    # mse_date_gpd = df_lst.groupby('date_time')
    # anomaly_sample = mse_date_gpd.get_group('201707251537')
    #
    # plt.plot(anomaly_sample.yTrue.values,label='True');plt.plot(anomaly_sample.yPred.values,label='Pred')
    # plt.xticks(fontsize=20);plt.yticks(fontsize=20);
    # plt.legend(fontsize=20);
    # plt.show()

r_squared_df =pd.DataFrame([(k[:-1], v[0], v[1]) for d in r_squared_list for k, v in d.items()],
                  columns=["IntraSession","mean_date", "mean_date_time"])
