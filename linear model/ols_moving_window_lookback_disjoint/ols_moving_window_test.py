import pandas as pd
from os import listdir;from os.path import isfile, join
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt; plt.rcParams["figure.figsize"] = (10,6)

'''transform'''
path = "/home/kanli/forth/";
data_path = path + "out_disjoint5_numba/"
# data_path = path + "out_ols_disjoint1
#
#
#
# 5_numba/"
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
df_lst = pd.concat(map(lambda file: pd.read_pickle(data_path + file), onlyfiles))
# groupby date /option 01
mean_date = df_lst.groupby('date').mean()
mse_date = df_lst.groupby('date').apply(lambda x: mean_squared_error(x['yTrue'],x['yPred']))
# groupby date and time /option 02
df_lst['date_time'] = df_lst.date.apply(str) + df_lst.timeHMs.apply(lambda x: str(x).zfill(4))
mean_date_time = df_lst.groupby('date_time').mean()
mse_date_time = df_lst.groupby('date_time').apply(lambda x: mean_squared_error(x['yTrue'],x['yPred']))

'''anomaly handling'''
mse_date_gpd = df_lst.groupby('date_time')
anomaly_sample = mse_date_gpd.get_group('201707251537')

'''plot'''
# groupby date /option 01
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

# groupby date and time /option 02
plt.plot(mean_date_time.yTrue.values,label='True');plt.plot(mean_date_time.yPred.values,label='Pred')
plt.xticks(fontsize=20);plt.yticks(fontsize=20);
# plt.legend();
plt.title("ols_disjoint5_yTrue_&_yPred")
plt.legend(fontsize=20);
plt.show()

plt.plot(mse_date_time.values,label="MSE");
plt.xticks(fontsize=20);plt.yticks(fontsize=20);
plt.legend(fontsize=20);
plt.title("ols_disjoint5_MSE")
# plt.legend();
plt.show()

# anomaly /option 03
plt.plot(anomaly_sample.yTrue.values,label='True');plt.plot(anomaly_sample.yPred.values,label='Pred')
plt.xticks(fontsize=20);plt.yticks(fontsize=20);
plt.legend(fontsize=20);
plt.show()


