import pandas as pd
from os import listdir;from os.path import isfile, join
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt; plt.rcParams["figure.figsize"] = (40,20)

'''transform'''
path = "/home/kanli/forth/"; data_path, out_path = path + "out_ols/", path + 'out_ols/'
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
df_lst = pd.concat(map(lambda file: pd.read_csv(data_path + file, index_col=0), onlyfiles))
# groupby date
mean_date = df_lst.groupby('date').mean()
mse_date = df_lst.groupby('date').apply(lambda x: mean_squared_error(x['yTrue'],x['yPred']))
# groupby date and time
df_lst['date_time'] = df_lst.date.apply(str) + df_lst.timeHMs.apply(lambda x: str(x).zfill(4))
mean_date_time = df_lst.groupby('date_time').mean()
mse_date_time = df_lst.groupby('date_time').apply(lambda x: mean_squared_error(x['yTrue'],x['yPred']))

'''anomaly handling'''
mse_date_gpd = df_lst.groupby('date_time')
anomaly_sample = mse_date_gpd.get_group('201707251537')

'''plot'''
plt.plot(mean_date.yTrue.values,label='True');plt.plot(mean_date.yPred.values,label='Pred')
# plt.xticks(fontsize=20);plt.yticks(fontsize=20);
plt.legend();
# plt.legend(fontsize=20);
plt.show()

plt.plot(mse_date.values,label="MSE");
# plt.xticks(fontsize=20);plt.yticks(fontsize=20);
# plt.legend(fontsize=20);
plt.legend();
plt.show()

plt.plot(mean_date_time.yTrue.values,label='True');plt.plot(mean_date_time.yPred.values,label='Pred')
plt.xticks(fontsize=20);plt.yticks(fontsize=20);
plt.legend();
plt.legend(fontsize=20);
plt.show()

plt.plot(mse_date_time.values,label="MSE");
plt.xticks(fontsize=20);plt.yticks(fontsize=20);
plt.legend(fontsize=20);
# plt.legend();
plt.show()

plt.plot(anomaly_sample.yTrue.values,label='True');plt.plot(anomaly_sample.yPred.values,label='Pred')
plt.xticks(fontsize=20);plt.yticks(fontsize=20);
plt.legend(fontsize=20);
plt.show()


