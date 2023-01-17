import numpy as np
import pandas as pd

from data_processing.utils import timestamp_format
from data_processing.get_data import get_message_data, get_orderbook_data
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


#
# if __name__ == "__main__":
#     pass
index, symbol = 0, "210426"
window_size = 1
message = get_message_data(index)
orderbook_data = get_orderbook_data(index)

merged_message = pd.merge(message, orderbook_data, how="left")
merged_message = timestamp_format(merged_message)
groupped_message = split_into_bucket(merged_message)
# plot_single_value(groupped_quantity.values)
features = get_basic_features(groupped_message, window_size)


