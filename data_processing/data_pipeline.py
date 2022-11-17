# %%
import numpy as np
import pandas as pd 
def get_message_data():
    # data_path = "/home/kanli/Volume-Transformer-main/AMZN_2021-04-01_34200000_57600000_message_50.csv"
    data_path = "/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_50.csv"
    df = pd.read_csv(data_path)
    message = df[df.iloc[:,1] == 4];message.columns = ['time','type','order_id','quantity','price','side','remark']
    message = message.drop(['remark'],axis = 1)
    return message

def get_orderbook_data():
    # data_path = "/home/kanli/Volume-Transformer-main/AMZN_2021-04-01_34200000_57600000_orderbook_50.csv";df = pd.read_csv(data_path)
    # data_path = "/Users/kang/Desktop/Volume-Tranformer/AMZN_2021-04-01_34200000_57600000_orderbook_50.csv";df = pd.read_csv(data_path)
    # df1 = df.iloc[:,[0,2]];df1.columns =['best_ask','best_bid'];df1
    # df2 = pd.merge(message, df1,left_index=True, right_index=True);df2 = df2.reset_index();df2 = df2.drop(['remark'],axis = 1);df2
    pass

def timestamp_format(message):
    message.time *= 1000000000
    message.time = message.time.astype("datetime64[ns]")
    message = message.reset_index()
    message = message.set_index('time')
    message = message.drop(['index'],axis = 1)
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

if __name__=="__main__":
    message = get_message_data()
    message = timestamp_format(message)
    groupped_message, groupped_quantity = split_into_bucket(message)
    plot_single_value(groupped_quantity.values)
    
    
    # return 
    # for item in groupped_message:
    #     print(item)
    #     break
    # message.resample('1min')

    # new = cut_tail(groupped_quantity)
    # plot_single_value(new.values)
    # top_quantity = message.quantity.quantile(0.95)
    # bottom_quantity = message.quantity.quantile(0.05)
