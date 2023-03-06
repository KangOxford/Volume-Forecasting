import pandas as pd
from os import listdir;from os.path import isfile, join
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt; plt.rcParams["figure.figsize"] = (12, 8)

'''transform'''
path = "/home/kanli/fifth/"
data_path = path + "out_1min/"
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
df_lst = pd.concat(map(lambda file: pd.read_pickle(data_path + file), onlyfiles))
# groupby date
mean_date = df_lst.groupby('date').mean()
mse_date = df_lst.groupby('date').apply(lambda x: mean_squared_error(x['yTrue'],x['yPred']))
# groupby date and time
df_lst['date_time'] = df_lst.date.apply(str) + df_lst.timeHMs.apply(lambda x: str(x).zfill(4))
mean_date_time = df_lst.groupby('date_time').mean()
mse_date_time = df_lst.groupby('date_time').apply(lambda x: mean_squared_error(x['yTrue'],x['yPred']))

# '''anomaly handling'''
# mse_date_gpd = df_lst.groupby('date_time')
# anomaly_sample = mse_date_gpd.get_group('201707251537')

def plot_pred_true(value,title, size=(12, 8)):
    plt.rcParams["figure.figsize"] = size
    plt.plot(value.yTrue.values, label='True')
    plt.plot(value.yPred.values, label='Pred')
    plt.title(title)
    plt.legend();
    plt.savefig(path + title +'.png')
    plt.show()
'''plot'''
'''date'''
plot_pred_true(mean_date,"yPred_&_yTrue ols1min mean_by_date")
plt.plot(mean_date.yTrue.values,label='True');plt.plot(mean_date.yPred.values,label='Pred')
# plt.xticks(fontsize=20);plt.yticks(fontsize=20);
title = "yPred_&_yTrue ols1min mean_by_date"
plt.title(title)
plt.legend();
# plt.legend(fontsize=20);
plt.show()

plt.plot(mse_date.values,label="MSE");
# plt.xticks(fontsize=20);plt.yticks(fontsize=20);
# plt.legend(fontsize=20);
plt.title("yPred_&_yTrue ols1min mse")
plt.legend();
plt.show()

'''date_time'''
plt.rcParams["figure.figsize"] = (40, 10)
plt.plot(mean_date_time.yTrue.values,label='True');plt.plot(mean_date_time.yPred.values,label='Pred')
# plt.xticks(fontsize=20);plt.yticks(fontsize=20);
title = "yPred_&_yTrue ols1min mean_by_date_time"
plt.title(title)
plt.legend();
# plt.legend(fontsize=20);
plt.savefig(path + title+".png")
plt.show()

plt.rcParams["figure.figsize"] = (12, 8)
plt.plot(mse_date_time.values,label="MSE");
plt.title("yPred_&_yTrue ols1min mean_by_date_time")
# plt.xticks(fontsize=20);plt.yticks(fontsize=20);
# plt.legend(fontsize=20);
plt.legend();
plt.show()

plt.plot(anomaly_sample.yTrue.values,label='True');plt.plot(anomaly_sample.yPred.values,label='Pred')
plt.xticks(fontsize=20);plt.yticks(fontsize=20);
plt.legend(fontsize=20);
plt.show()


