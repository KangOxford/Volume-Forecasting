import pandas as pd
import math
import numpy as np
from datetime import datetime, timedelta, date
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
# %matplotlib inline
#
import warnings
warnings.filterwarnings(action='ignore')

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


# ======================
rws = 3 # rolling_window_size
daily_Stats_10 = dict()
for sym_name in syms:
    # sym_name = syms[0]
    mask = df_clean_10.sym == sym_name
    temp = df_clean_10[mask]
    daily_vol = temp["volume"].groupby(temp['date']).sum()

    daily_log_vol = np.log(daily_vol)
    log_mean = daily_log_vol.rolling(rws).mean()
    mean = daily_vol.rolling(rws).mean()
    std = daily_vol.rolling(rws).std()

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

    cum_vol_prof_var = temp.groupby('time')["cum_vol_prof"].apply(lambda x: x.rolling(rws).var().shift(1))
    vol_prof_mean = temp.groupby('time')["vol_prof"].apply(lambda x: x.rolling(rws).mean().shift(1))
    cum_vol_prof_mean = temp.groupby('time')["cum_vol_prof"].apply(lambda x: x.rolling(rws).mean().shift(1))

    df_temp = pd.DataFrame()
    df_temp["cum_vol_prof_var"] = cum_vol_prof_var
    df_temp["vol_prof_mean"] = vol_prof_mean
    df_temp["cum_vol_prof_mean"] = cum_vol_prof_mean

    minutes_Stats_10.append(df_temp)

df_minutes_Stats_10 = pd.concat(minutes_Stats_10)
df_clean_10 = df_clean_10.join(df_minutes_Stats_10)

def TimeDiff(time):
    return (datetime.combine(date.min, time) - datetime.combine(date.min, datetime.strptime('09:30', '%H:%M').time())).seconds/60
df_clean_10["time_diff"] = df_clean_10["time"].copy().apply(TimeDiff)

df_clean_10.info()
df_cleaned_10 = df_clean_10.dropna()
# df_cleaned_10.to_pickle("df_clean_10.pkl")
# df_cleaned_10 = pd.read_pickle("df_clean_10.pkl")

# ============================
def DtermBlend(m, c, df, ADV_num):
    days = list(set(df["date"]))
    days.sort()

    for day in days[ADV_num:]:
        mask1 = (df.date == day) & (df.date_time <= (df[df.date == day].date_time.iloc[0] + timedelta(minutes=m)))
        mask2 = (df.date == day) & (df.date_time >= (df[df.date == day].date_time.iloc[0] + timedelta(minutes=m)))

        ADV = df.loc[
            (df.date == day) & (df.date_time == (df[df.date == day].date_time.iloc[0])), "daily_vol_mean"].values
        df.loc[mask1, "Deter_Blend"] = (1 - df.loc[mask1, "time_diff"] * c / m) * ADV + df.loc[
            mask1, "time_diff"] * c / m * df.loc[mask1, "cum_sum_vol"] / df.loc[mask1, "cum_vol_prof_mean"]
        df.loc[mask2, "Deter_Blend"] = (1 - (c + (1 - c) * (df.loc[mask2, "time_diff"] - m) / (390 - m))) * ADV + (
                    c + (1 - c) * (df.loc[mask2, "time_diff"] - m) / (390 - m)) * df.loc[mask2, "cum_sum_vol"] / df.loc[
                                           mask2, "cum_vol_prof_mean"]

    return df

Dterm_list_10 = []
for sym_name in syms:
    # sym_name = syms[0]
    mask = df_clean_10.sym == sym_name
    temp = DtermBlend(30, 1/3, df_clean_10[mask], ADV_num = rws)
    temp["Deter_Blend_interval_vol"] = temp.groupby('date').apply(lambda x :x.Deter_Blend.shift(1)*x.vol_prof_mean).values
    temp.loc[temp.time==datetime.strptime('09:40', '%H:%M').time(),"Deter_Blend_interval_vol"]  = temp[temp.time==datetime.strptime('09:40', '%H:%M').time()].daily_vol_mean*temp[temp.time==datetime.strptime('09:40', '%H:%M').time()].vol_prof_mean
    Dterm_list_10.append(temp)

df_Dterm_10 = pd.concat(Dterm_list_10)

# =================

# For whole data set & for random continuous 5 days for each sym

import random
import matplotlib.pyplot as plt

for sym_name in syms:
    # sym_name = syms[0]
    mask = df_Dterm_10.sym == sym_name
    temp = df_Dterm_10[mask].dropna().set_index("date_time")
    temp.index = temp.index.to_series().apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
    index = random.randrange(0, 4095)
    # temp[["cum_sum_vol", "daily_vol", "Deter_Blend", "daily_vol_mean"]].plot(figsize=(18, 8), title=sym_name + " 10-min")
    # plt.xlabel("Time")
    # plt.ylabel("Volume")
    plt.show()

    rand_temp = temp.iloc[index:index + 5 * 39]
    # rand_temp[["cum_sum_vol", "daily_vol", "Deter_Blend", "daily_vol_mean"]].plot(figsize=(18, 8), title=sym_name + " 10-min")
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.show()

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.scatter(rand_temp.index, rand_temp["Deter_Blend_interval_vol"], label="Deter_Blend_interval_vol")
    ax.scatter(rand_temp.index, rand_temp["volume"], label="realized volume")
    xpositions = rand_temp[rand_temp.time == datetime.strptime('09:40', '%H:%M').time()].index.values
    for xc in xpositions:
        plt.axvline(x=xc, color='k', linestyle='--')
    # ax.set_xticks(rand_temp.index[::39])
    # ax.set_title(sym_name + " 10-min" + " Intraday volume distribution")
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Volume")
    plt.show()









































