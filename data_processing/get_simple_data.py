import numpy as np
import pandas as pd
import datetime

from data_processing.utils import timestamp_format
from data_processing.get_data import DataPath
# from data_processing.get_data import get_message_data, get_orderbook_data
from data_processing.utils import split_into_bucket, get_num_vol_ntn, time_index_formatting, window


def get_basic_features(groupped_message, window_size=1):
    w = window(groupped_message, window_size)
    signal_list = []
    for next_w in w:
        # ----------------- 01 -----------------
        # next_w = next(w)
        list_ = [item[1] for item in next_w]
        item = pd.concat(list_)
        time_index_start = next_w[0][0]
        time_index_end = next_w[-1][0]
        # ----------------- 01 -----------------
        signal = {}
        time_index_start = time_index_formatting(time_index_start)
        time_index_end = time_index_formatting(time_index_end)
        signal['timeHM_start'] = str(time_index_start[0]).zfill(2) + str(time_index_start[1]).zfill(2)
        signal['timeHM_end'] = str(time_index_end[0]).zfill(2) + str(time_index_end[1]).zfill(2)
        # ----------------- 02 -----------------
        signal = get_num_vol_ntn(item, signal)
        # # --------------- 04 -----------------
        signal['volume'] = item.quantity.sum()
        # ----------------- 05 -----------------
        signal_list.append(signal)
        # print() #$
    features = pd.DataFrame(signal_list)
    features['timeHM_end'] = features.timeHM_end.shift(-1);
    features.timeHM_end.iloc[-1] = '1600'
    return features


def get_simple_features(groupped_message, window_size=1):
    w = window(groupped_message, window_size)
    signal_list = []
    for next_w in w:
        # ----------------- 01 -----------------
        list_ = [item[1] for item in next_w]
        item = pd.concat(list_)
        time_index_start = next_w[0][0]
        time_index_end = next_w[-1][0]
        # ----------------- 01 -----------------
        signal = {}
        time_index_start = time_index_formatting(time_index_start)
        time_index_end = time_index_formatting(time_index_end)
        signal['timeHM_start'] = str(time_index_start[0]).zfill(2) + str(time_index_start[1]).zfill(2)
        # ----------------- 02 -----------------
        signal = get_num_vol_ntn(item, signal)
        # # --------------- 04 -----------------
        signal['volume'] = item.quantity.sum()
        # ----------------- 05 -----------------
        signal_list.append(signal)
        # print() #$
    features = pd.DataFrame(signal_list)
    return features


def get_volume(groupped_message, window_size=1):
    w = window(groupped_message, window_size)
    signal_list = []
    for next_w in w:
        # ----------------- 01 -----------------
        list_ = [item[1] for item in next_w]
        item = pd.concat(list_)
        time_index_start = next_w[0][0]
        # ----------------- 01 -----------------
        signal = {}
        time_index_start = time_index_formatting(time_index_start)
        signal['time'] = str(time_index_start[0]).zfill(2) + str(time_index_start[1]).zfill(2)
        # signal['timeHM_start'] = str(time_index_start[0]).zfill(2) + str(time_index_start[1]).zfill(2)
        # # --------------- 04 -----------------
        signal['volume'] = item.quantity.sum()
        # ----------------- 05 -----------------
        signal_list.append(signal)
        # print() #$
    features = pd.DataFrame(signal_list)
    return features

def get_simple_volume(datapath, index):
    message = datapath.get_message_data(index)
    orderbook_data = datapath.get_orderbook_data(index)
    merged_message = pd.merge(message, orderbook_data, how="left")
    merged_message = timestamp_format(merged_message)
    groupped_message = split_into_bucket(merged_message)
    features = get_volume(groupped_message, window_size)
    return features

def get_wrapped_volume(datapath, index):
    features = get_simple_volume(datapath, index)
    date = datapath.level10[index][0].split('_')[1]
    sym = datapath.level10[index][0].split('_')[0]
    new_features = features.copy()
    new_features['date'] = datetime.date(int(date.split('-')[0]), \
                                         int(date.split('-')[1]), \
                                         int(date.split('-')[2]))
    new_features['sym'] = sym
    new_features.time = new_features.time.apply(lambda x:
                            datetime.time(int(x[:2]), \
                                          int(x[2:])))
    rst = new_features[['date','time','sym','volume']]
    return rst

def get_monthly_volume():
    # window_size = 1
    datapath = DataPath()
    # datapath_size = len(datapath.level10)
    features_list = []
    for index in range(len(datapath.level10)):
        # index = 1
        # print(index)
        features = get_wrapped_volume(datapath, index)
        features_list.append(features)
    concatted_features = pd.concat(features_list)
    concatted_features = concatted_features.reset_index()
    return concatted_features

if __name__ == "__main__":
    features = get_monthly_volume()
    features.to_csv("features.csv")
    features.to_pickle("features.pkl")

    # import matplotlib.pyplot as plt
    # plt.figure(dpi=600, figsize=(20, 6))
    # features.volume.plot()
    # plt.show()


