import pandas as pd
import os;from os import listdir;from os.path import isfile, join
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt; plt.rcParams["figure.figsize"] = (12, 8)
from sklearn.metrics import r2_score

'''transform'''
import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
    data_path = path + "03_out_15min_pred_true_pairs_after_ols/"
    out_path = path + "04_pred_true_fig/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    path = "/home/kanli/fifth/"
    data_path = path + "out_15min_pred_true_pairs_after_ols/"
    out_path = path + "04_pred_true_fig/"
else:print("Unknown operating system")

try: listdir(out_path)
except:import os;os.mkdir(out_path)

onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
df_lst = pd.concat(map(lambda file: pd.read_pickle(data_path + file), onlyfiles))
# groupby date
mean_date = df_lst.groupby('date').mean()
r_squared_mean_date = r2_score(mean_date['yTrue'], mean_date['yPred'])
mse_date = df_lst.groupby('date').apply(lambda x: mean_squared_error(x['yTrue'],x['yPred']))
# groupby date and time
df_lst['date_time'] = df_lst.date.apply(str) + df_lst.timeHMs.apply(lambda x: str(x).zfill(4))
mean_date_time = df_lst.groupby('date_time').mean()
r_squared_mean_date_time = r2_score(mean_date_time['yTrue'], mean_date_time['yPred'])
mse_date_time = df_lst.groupby('date_time').apply(lambda x: mean_squared_error(x['yTrue'],x['yPred']))



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


