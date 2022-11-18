# %%
import numpy as np
import pandas as pd 
class Config:
    scale_level = 10000
def get_message_data():
    # data_path = "/home/kanli/Volume-Transformer-main/AMZN_2021-04-01_34200000_57600000_message_50.csv"
    data_path = "/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_50.csv"
    df = pd.read_csv(data_path)
    message = df[df.iloc[:,1] == 4];message.columns = ['time','type','order_id','quantity','price','side','remark']
    message = message.drop(['remark'],axis = 1)
    message = message.reset_index()
    return message

def get_orderbook_data():
    # data_path = "/home/kanli/Volume-Transformer-main/AMZN_2021-04-01_34200000_57600000_orderbook_50.csv";df = pd.read_csv(data_path)
    data_path = "/Users/kang/Desktop/Volume-Tranformer/AMZN_2021-04-01_34200000_57600000_orderbook_50.csv"
    df = pd.read_csv(data_path)
    df1 = df.iloc[:,[0,2]];df1.columns =['best_ask','best_bid']
    df1['mid_price'] = (df1.best_ask + df1.best_bid)//2
    df1 = df1.reset_index()
    return df1

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
    
# def split_by_resample(message):
#     msg1 = message.resample('5min') 
#     return msg1    
    
def split_into_bucket(message, freq = '1min'):
    msg = message.reset_index()
    # groupped_message = msg.groupby(pd.Grouper(key = 'time', axis = 0, freq = '5min'))
    # groupped_message = msg.groupby(pd.Grouper(key = 'time', axis = 0, freq = freq))
    groupped_message = msg.groupby(pd.Grouper(key = 'time', axis = 0, freq = freq))
    # groupped_message = message.groupby([[d.hour for d in message.index],[d.minute for d in message.index]])
    return groupped_message

def cut_tail(groupped_quantity):
    top_quantity = groupped_quantity.quantile(0.95)
    btm_quantity = groupped_quantity.quantile(0.05)
    new = groupped_quantity[groupped_quantity <= top_quantity]
    new = new[new >= btm_quantity]
    return new 

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
        time_index_start = next_w[0][0] 
        time_index_end = next_w[-1][0] 
        # time_index, item = bigram[0], bigram[1]
        # ----------------- 01 -----------------
        signal = {}
        time_index_start = time_index_formatting(time_index_start)
        time_index_end = time_index_formatting(time_index_end)
        signal['timeHM_start'] = str(time_index_start[0]).zfill(2) + str(time_index_start[1]).zfill(2)
        signal['timeHM_end'] = str(time_index_end[0]).zfill(2) + str(time_index_end[1]).zfill(2)
        x_bid = sum(item.side == 1); x_ask = sum(item.side == -1)
        signal['imbalance'] = (x_bid - x_ask)/(x_bid + x_ask)
        
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


# def feature_disjoint(windows= [1,5]):
#     level1,level2 = windows[0], windows[1]
#     features1, _ = get_data(level1)
#     features5, _ = get_data(level2)
#     features1_new = features1.shift(-level2)
#     features1_n = features1_new.dropna(axis = 0)
#     features5_new = features5.drop(['target'],axis =1)
#     features5_n = features5_new.iloc[:-level1,:]
#     features5_n.columns = ["5_"+item for item in features5_n.columns]
#     features1_5 = pd.concat([features5_n , features1_n],axis=1)
#     return features1_5

def feature_overlap(windows= [1,5]):
    level1,level2 = windows[0], windows[1]
    features1, _ = get_data(level1)
    features5, _ = get_data(level2)
    # ----------------- 01 -----------------
    # breakpoint()
    features1_new = features1.shift(-level2+1)
    features1_n = features1_new.dropna(axis = 0)
    # ----------------- 01 -----------------
    features5_n = features5.drop(['target'],axis =1)
    features5_n.columns = ["5_"+item for item in features5_n.columns]
    # ----------------- 01 -----------------
    features1_5 = pd.concat([features5_n , features1_n],axis=1)
    return features1_5

# def feature_overlap(windows= [1,5,10]):
#     level1,level2,level3 = windows[0], windows[1],windows[2]
#     features1, _ = get_data(level1)
#     features5, _ = get_data(level2)
#     features10, _ = get_data(level3)
#     # ----------------- 01 -----------------
#     features1_new = features1.shift(-level3+1)
#     features1_n = features1_new.dropna(axis = 0)
#     # ----------------- 01 -----------------
#     features5_new = features5.drop(['target'],axis =1)
#     features5_n = features5_new.shift(-(level3-level2))
#     features5_n = features5_n.dropna(axis = 0)
#     # features5_n = features5_new.iloc[:-level1,:]
#     features5_n.columns = ["5_"+item for item in features5_n.columns]
#     # ----------------- 01 -----------------
#     features10_n = features10.drop(['target'],axis =1)
#     features10_n.columns = ["10_"+item for item in features10_n.columns]
#     # ----------------- 01 -----------------
#     features1_5_10 = pd.concat([features10_n, features5_n , features1_n],axis=1)
#     return features1_5_10

# def feature_disjoint(windows= [1,5,10]):
#     level1,level2,level3 = windows[0], windows[1],windows[2]
#     features1, _ = get_data(level1)
#     features5, _ = get_data(level2)
#     features10, _ = get_data(level3)
#     # ----------------- 01 -----------------
#     # breakpoint()
#     features1_new = features1.shift(-level3-level2)
#     features1_n = features1_new.dropna(axis = 0)
#     features1_n = features1_n.reset_index().drop(['index'],axis =1)
#     # ----------------- 01 -----------------
#     features5_new = features5.drop(['target'],axis =1)
#     features5_n = features5_new.shift(-level3)
#     features5_n = features5_n.dropna(axis = 0)
#     # features5_n = features5_new.iloc[:-level1,:]
#     features5_n.columns = ["5_"+item for item in features5_n.columns]
#     features5_n = features5_n.shift(level1).dropna()
#     features5_n = features5_n.reset_index().drop(['index'],axis =1)
#     # ----------------- 01 -----------------
#     features10_n = features10.drop(['target'],axis =1)
#     features10_n.columns = ["10_"+item for item in features10_n.columns]
#     features10_n = features10_n.shift(level2+level1).dropna()
#     features10_n = features10_n.reset_index().drop(['index'],axis =1)
#     # ----------------- 01 -----------------
#     features1_5_10 = pd.concat([features10_n, features5_n , features1_n],axis=1)
#     return features1_5_10
    
    
if __name__=="__main__":
    # features1_5_10 = feature_overlap()
    # features1_5_10 = feature_disjoint()
    features1_5 = feature_overlap()
    
    
    
    
    
    
    
    
    
    
    
    
    