# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 

from data_processing.config import Config
from data_processing.utils import timestamp_format
from data_processing.get_data import get_message_data,get_orderbook_data
    
def split_into_bucket(message, freq = '1D'):
    msg = message.reset_index()
    # groupped_message = msg.groupby(pd.Grouper(key = 'time', axis = 0, freq = '5min'))
    # groupped_message = msg.groupby(pd.Grouper(key = 'time', axis = 0, freq = freq))
    groupped_message = msg.groupby(pd.Grouper(key = 'time', axis = 0, freq = freq))
    # groupped_message = message.groupby([[d.hour for d in message.index],[d.minute for d in message.index]])
    return groupped_message

def get_basic_features(groupped_message, window_size = 1):
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
    w = window(groupped_message, window_size)  
    signal_list = []
    for next_w in w:
        # ----------------- 01 -----------------
        # next_w = next(w)
        list_ = [item[1] for item in next_w]
        item = pd.concat(list_)  
        # ----------------- 01 -----------------
        signal = {}
        
        from datetime import datetime
        date = item.time[0].date()
        open_time = '10:00:00'
        open_session = datetime.strptime(str(date)+"-"+open_time, '%Y-%m-%d-%H:%M:%S')
        # datetime.timestamp(open_session)
        
        
        # time = item.time.map(lambda x: x.to_pydatetime())
        # new_t = item.time.map(lambda x: x.to_pydatetime())
        date_time = []
        for i in range(item.time.shape[0]):
            t = item.time[i].to_pydatetime()
            date_time.append(t)
        date_time = np.array(date_time)    
        open_ = date_time < open_session
        item['session'] = open_.astype(np.int)
        
        
        
        
        # item['session'] =
        # date_time_series = pd.Series(date_time)
        # item['datetime'] = date_time
        
        
        
        
        
        
        if '0930'<= signal['timeHM_start'] and signal['timeHM_start'] <='1000':
            signal['intrady_session'] = 0
        elif '1530'<= signal['timeHM_start'] and signal['timeHM_start'] <='1600':
            signal['intrady_session'] = 2
        else:
            signal['intrady_session'] = 1
        # ----------------- 02 -----------------
        signal = get_num_vol_ntn(item, signal)
        # ----------------- 03 -----------------
        item['aggressive'] = ((item.price >= item.mid_price) & (item.side == 1)) | ((item.price <= item.mid_price) & (item.side == -1))
        aggressive = item[item.aggressive]
        signal = get_num_vol_ntn(aggressive, signal, sym = 'ag_')
        # # ----------------- 04 -----------------
        # signal['volume'] = item.quantity.sum()
        # ----------------- 05 -----------------
        signal_list.append(signal)
        # print() #$
    features = pd.DataFrame(signal_list)
    features['timeHM_end'] = features.timeHM_end.shift(-1); features.timeHM_end.iloc[-1] = '1600'
    return features

def get_data(window_size =1):
    message = get_message_data()

    orderbook_data = get_orderbook_data()
    merged_message = pd.merge(message, orderbook_data, how = "left")
    merged_message = timestamp_format(merged_message)
    groupped_message = split_into_bucket(merged_message)
    # plot_single_value(groupped_quantity.values)
    features = get_basic_features(groupped_message,window_size)
    features['volume'] = features.bid_volume + features.ask_volume
    features['vol_change'] = features.volume.diff()/features.volume
    features['vol_direction'] = features.vol_change.apply(lambda x: -1 if x<= 0 else 1)
    features.timeHM_start = features.timeHM_start.apply(lambda x: int(x[0:2]) + int(x[2:])*0.01)
    features.timeHM_end = features.timeHM_end.apply(lambda x: int(x[0:2]) + int(x[2:])*0.01)
    features['target'] = features['volume'].shift(-1)
    return features.dropna(),features   


if __name__=="__main__":
    features, _ = get_data()