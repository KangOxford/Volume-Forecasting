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
    
if __name__=="__main__":
    message = get_message_data()
    def timestamp_format(message):
        
        return message
    message.time.astype("datetime64[s]")
    # def split_into_bucket(message):
    # from datetime import datetime
    # datetime.now()
    from dateutil.parser import parse
    parse("34201.238593")
    pd.to_datetime("34201.238593")
    message.time *= 1000000000
    message.time.astype("datetime64[ns]")
    # def cut_tail(message):
    # top_quantity = message.quantity.quantile(0.95)
    # bottom_quantity = message.quantity.quantile(0.05)
