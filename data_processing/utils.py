# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from data_processing.config import Config

def split_into_bucket(message, freq='1min'):
    msg = message.reset_index()
    # groupped_message = msg.groupby(pd.Grouper(key = 'time', axis = 0, freq = '5min'))
    # groupped_message = msg.groupby(pd.Grouper(key = 'time', axis = 0, freq = freq))
    groupped_message = msg.groupby(pd.Grouper(key='time', axis=0, freq=freq))
    # groupped_message = message.groupby([[d.hour for d in message.index],[d.minute for d in message.index]])
    return groupped_message


def cut_tail(groupped_quantity):
    top_quantity = groupped_quantity.quantile(0.95)
    btm_quantity = groupped_quantity.quantile(0.05)
    new = groupped_quantity[groupped_quantity <= top_quantity]
    new = new[new >= btm_quantity]
    return new

def timestamp_format(message):
    message.time *= 1000000000
    message.time = message.time.astype("datetime64[ns]")
    message = message.reset_index()
    message = message.set_index('time')
    message = message.drop(['index','level_0'],axis = 1)
    return message
    
def plot_single_value(value):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    figure(figsize=(50, 10), dpi=80)
    plt.plot(value)


def get_num_vol_ntn(item, signal, sym = ''):
    bid_item = item[item.side == 1]; ask_item = item[item.side == -1]
    signal[sym + 'bid_num_orders'] = bid_item.shape[0]; signal[sym+'ask_num_orders'] = ask_item.shape[0]
    signal[sym+'bid_volume'] = bid_item.quantity.sum(); signal[sym+'ask_volume'] = ask_item.quantity.sum()
    signal[sym+'bid_notional']=(bid_item.quantity * bid_item.price).sum()/Config.scale_level
    signal[sym+'ask_notional']=(ask_item.quantity * ask_item.price).sum()/Config.scale_level
    return signal
def time_index_formatting(time_index):
    if type(time_index) == pd._libs.tslibs.timestamps.Timestamp:
        string_ = str(time_index)
        time_index = (string_[-8:-6], string_[-5:-3])
    return time_index

def window(seq, n=1):
    from itertools import islice
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
