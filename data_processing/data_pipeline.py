# %%
import numpy as np
import pandas as pd 

data_path = "/home/kanli/Volume-Transformer-main/AMZN_2021-04-01_34200000_57600000_message_50.csv"
# data_path = "/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_50.csv"
df = pd.read_csv(data_path)
message = df[df.iloc[:,1] == 4];message.columns = ['time','type','order_id','quantity','price','side','remark']
data_path = "/home/kanli/Volume-Transformer-main/AMZN_2021-04-01_34200000_57600000_orderbook_50.csv";df = pd.read_csv(data_path)
# data_path = "/Users/kang/Desktop/Volume-Tranformer/AMZN_2021-04-01_34200000_57600000_orderbook_50.csv";df = pd.read_csv(data_path)
df1 = df.iloc[:,[0,2]];df1.columns =['best_ask','best_bid'];df1
df2 = pd.merge(message, df1,left_index=True, right_index=True);df2 = df2.reset_index();df2 = df2.drop(['remark'],axis = 1);df2
# %%
df2
# %%
