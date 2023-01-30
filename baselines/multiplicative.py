import os
import pandas as pd
import math
import numpy as np
from datetime import datetime, timedelta, date
import time
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
import pickle
import datetime
# df = pd.read_csv("/Users/kang/Desktop/Volume-Forecasting/features.csv")[["date","time","sym","volume"]]
# df = pd.read_pickle("/Users/kang/Desktop/Volume-Forecasting/features.pkl")
df = pd.read_pickle(os.environ['HOME'] + "/Volume-Forecasting/features.pkl")
# import os;os.system("echo $HOME/Volume-Forecasting/")

df = df[df.date != datetime.date(2021,4,12)]
df.volume = df.volume+1
df['Index'] = df.index
df.index = df.index+1


# fig, ax = plt.subplots(figsize=(100, 8))
# ax.scatter(df.index, df["volume"])
# xpositions = df[df.time == datetime.strptime('09:40', '%H:%M').time()].index.values
# for xc in xpositions:
#     plt.axvline(x=xc, color='k', linestyle='--')
# # ax.set_xticks(rand_temp.index[::39])
# # ax.set_title(sym_name + " 10-min" + " Intraday volume distribution")
# # ax.set_xlabel("Time")
# # ax.set_ylabel("Volume")
# plt.savefig("volume.png")
# plt.show()

# df['datetime'] = df['date'] + ' ' + df['time']
# df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
# new = df[['datetime', 'volume']]


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

austourists = df.volume
model = ETSModel(
    austourists,
    error="add",
    trend="add",
    seasonal="add",
    damped_trend=True,
    seasonal_periods=4,
)
fit = model.fit()


# ===================================== {
new = df[['Index', 'volume']]

from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(new.volume, period = 390, model="multiplicative")
# decompose_result = seasonal_decompose(new, period = 390, model="multiplicative")

#
# import statsmodels.api as sm
# res = sm.tsa.seasonal_decompose(new,freq=12,model="multiplicative")


trend = decompose_result.trend
seasonal = decompose_result.seasonal
residual = decompose_result.resid

from matplotlib.pyplot import figure
figure(figsize=(28, 6), dpi=80)
decompose_result.plot()
# decompose_result.plot(weights=(28, 6))
plt.show()
# ===================================== }
