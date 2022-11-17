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

def split_into_bucket(message):
    groupped_message = message.groupby([[d.hour for d in message.index],[d.minute for d in message.index]])
    groupped_quantity = message['quantity'].groupby([[d.hour for d in message.index],[d.minute for d in message.index]]).sum()
    return groupped_message, groupped_quantity

def cut_tail(groupped_quantity):
    top_quantity = groupped_quantity.quantile(0.95)
    btm_quantity = groupped_quantity.quantile(0.05)
    new = groupped_quantity[groupped_quantity <= top_quantity]
    new = new[new >= btm_quantity]
    return new 

def get_basic_features(groupped_message):
    def get_num_vol_ntn(item, signal, sym = ''):
        bid_item = item[item.side == 1]; ask_item = item[item.side == -1]
        signal[sym + 'bid_num_orders'] = bid_item.shape[0]; signal[sym+'ask_num_orders'] = ask_item.shape[0]
        signal[sym+'bid_volume'] = bid_item.quantity.sum(); signal[sym+'ask_volume'] = ask_item.quantity.sum()        
        signal[sym+'bid_notianal']=(bid_item.quantity * bid_item.price).sum()/Config.scale_level
        signal[sym+'ask_notianal']=(ask_item.quantity * ask_item.price).sum()/Config.scale_level
        return signal
    signal_list = []
    for time_index, item in groupped_message:
        # ----------------- 01 -----------------
        signal = {}
        signal['timeHM_start'] = str(time_index[0]).zfill(2) + str(time_index[1]).zfill(2)
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
    features['timeHM_end'] = features.timeHM_start.shift(-1); features.timeHM_end.iloc[-1] = '1600'
    return features

if __name__=="__main__":
    message = get_message_data()

    orderbook_data = get_orderbook_data()
    merged_message = pd.merge(message, orderbook_data, how = "left")
    merged_message = timestamp_format(merged_message)
    groupped_message, groupped_quantity = split_into_bucket(merged_message)
    # plot_single_value(groupped_quantity.values)
    features = get_basic_features(groupped_message)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    