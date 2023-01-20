import pandas as pd
import math
import numpy as np
from datetime import datetime, timedelta, date
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
# %matplotlib inline
df = pd.read_csv("/Users/kang/Desktop/Volume-Forecasting/features.csv")[["date","time","sym","volume"]]

all_time = list(set(df["time"]))
all_date = list(set(df["date"]))
all_date.sort()
all_time.sort()
print("Length of Days = %f" %len(all_date))
print("Length of Times = %f" %len(all_time))
all_date_time = [x+' '+ y for x in all_date for y in all_time]
all_date_time.sort()
len(all_date_time) == 18*390


df_full = pd.DataFrame(columns=["date","time","sym","date_time"])
df_full["date_time"] = all_date_time*1
new = df_full["date_time"].str.split(" ", n = 1, expand = True)
df_full.loc[:,"date"] = new[0]
df_full.loc[:,"time"] = new[1]
syms = df["sym"].unique()

for k in range(1):
    df_full.loc[k*18*390:(k+1)*18*390,"sym"] = syms[k]



df_clean = pd.merge(df_full,df,how='left')
print("Missing time intervals: ", pd.isna(df_clean).sum())
for sym_name in syms:
    mask = df_clean.sym == sym_name
    temp = df_clean[mask]

    fig, ax = plt.subplots(figsize = (10, 8))
    sns.heatmap(temp.isnull())
    ax.set_title(sym_name)
    plt.show()
# df_clean = df_clean.dropna()
df_clean = df_clean[df_clean.date != '2021-04-12'].reset_index()
df_clean = df_clean.drop(['index'],axis = 1)
# df_clean = df.copy()
df_clean.loc[:,"date_time"] = pd.to_datetime(df_clean["date_time"])
df_clean.loc[:,"date"] = pd.to_datetime(df_clean["date"]).dt.date
df_clean.loc[:,"time"] = pd.to_datetime(df_clean["time"]).dt.time

all_time = list(set(df_clean["time"]))
all_date = list(set(df_clean["date"]))
all_date.sort()
all_time.sort()
df_clean_10 = df_clean.set_index("date_time")
df_clean_10 = df_clean_10.groupby('sym').resample('10min', loffset=pd.Timedelta('10min')).sum(min_count=1).reset_index()
df_clean_10["date"] = df_clean_10.date_time.dt.date
df_clean_10["time"] = df_clean_10.date_time.dt.time

from datetime import datetime, time
def is_time_between(check_time):
    begin_time, end_time = time(9,40), time(16,0)
    # If check time is not given
    check_time = check_time or datetime.utcnow().time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
df_clean_10["filter"] = df_clean_10.time.apply(is_time_between)
df_clean_10 = df_clean_10[df_clean_10["filter"] == True].dropna().reset_index()
df_clean_10 = df_clean_10.drop(["index","filter"],axis=1)


df_clean_10.loc[df_clean_10["volume"] == 0, "volume"] = 1
### Add 1 share to all missing intervals to avoid the log(0) issue
df_clean_10.info()


# ------------
daily_Stats_10 = dict()
for sym_name in syms:
    mask = df_clean_10.sym == sym_name
    temp = df_clean_10[mask]
    daily_vol = temp["volume"].groupby(temp['date']).sum()

    daily_log_vol = np.log(daily_vol)
    log_mean = daily_log_vol.rolling(20).mean()
    mean = daily_vol.rolling(20).mean()
    std = daily_vol.rolling(20).std()

    df_temp = pd.DataFrame()
    df_temp["daily_vol"] = daily_vol
    df_temp["daily_log_vol"] = daily_log_vol
    df_temp["daily_vol_mean"] = mean
    df_temp["daily_vol_std"] = std
    df_temp["daily_vol_log_mean"] = log_mean

    df_temp[["daily_vol_mean", "daily_vol_std", "daily_vol_log_mean"]] = df_temp[
        ["daily_vol_mean", "daily_vol_std", "daily_vol_log_mean"]].shift(1)

    daily_Stats_10[sym_name] = df_temp
# ------------
for sym_name in syms:
    for stat in ["daily_vol", "daily_vol_mean", "daily_vol_std", "daily_vol_log_mean"]:
        df_clean_10.loc[(df_clean_10.sym == sym_name), stat] = df_clean_10.loc[(df_clean_10.sym == sym_name), :][
            "date"].map(daily_Stats_10[sym_name][stat])

    mask = df_clean_10.sym == sym_name
    df_clean_10.loc[mask, "cum_sum_vol"] = df_clean_10.loc[mask, "volume"].groupby(df_clean_10['date']).cumsum()

df_clean_10.loc[:, "cum_vol_prof"] = (df_clean_10["cum_sum_vol"] / df_clean_10["daily_vol"]).astype("float")
df_clean_10.loc[:, "vol_prof"] = (df_clean_10["volume"] / df_clean_10["daily_vol"]).astype("float")


# ------------
# Calculate 10 Minute Stats
minutes_Stats_10 = list()
for sym_name in syms:
    # sym_name = syms[0]
    mask = df_clean_10.sym == sym_name
    temp = df_clean_10[mask]

    cum_vol_prof_var = temp.groupby('time')["cum_vol_prof"].apply(lambda x: x.rolling(20).var().shift(1))
    vol_prof_mean = temp.groupby('time')["vol_prof"].apply(lambda x: x.rolling(20).mean().shift(1))
    cum_vol_prof_mean = temp.groupby('time')["cum_vol_prof"].apply(lambda x: x.rolling(20).mean().shift(1))

    df_temp = pd.DataFrame()
    df_temp["cum_vol_prof_var"] = cum_vol_prof_var
    df_temp["vol_prof_mean"] = vol_prof_mean
    df_temp["cum_vol_prof_mean"] = cum_vol_prof_mean

    minutes_Stats_10.append(df_temp)










