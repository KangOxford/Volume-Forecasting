# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 



class DataPath:
    level50 = [
        ("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_50.csv",
         "/Users/kang/Desktop/Volume-Tranformer/AMZN_2021-04-01_34200000_57600000_orderbook_50.csv"),
        ]
    level10 = [
        ("/Users/kang/Data/AMZN_2021/AMZN_2021-04-26_34200000_57600000_message_10.csv",
         "/Users/kang/Data/AMZN_2021/AMZN_2021-04-26_34200000_57600000_orderbook_10.csv"),
        
        ("/Users/kang/Data/AMZN_2021/AMZN_2021-04-21_34200000_57600000_message_10.csv",
          "/Users/kang/Data/AMZN_2021/AMZN_2021-04-21_34200000_57600000_orderbook_10.csv"),
        
        ("/Users/kang/Data/AMZN_2021/AMZN_2021-04-20_34200000_57600000_message_10.csv",
          "/Users/kang/Data/AMZN_2021/AMZN_2021-04-20_34200000_57600000_orderbook_10.csv"),
        
        ("/Users/kang/Data/AMZN_2021/AMZN_2021-04-19_34200000_57600000_message_10.csv",
          "/Users/kang/Data/AMZN_2021/AMZN_2021-04-19_34200000_57600000_orderbook_10.csv"),

        ("/Users/kang/Data/AMZN_2021/AMZN_2021-04-15_34200000_57600000_message_10.csv",
          "/Users/kang/Data/AMZN_2021/AMZN_2021-04-15_34200000_57600000_orderbook_10.csv")
        ]
        

def get_message_data(index):
    # data_path = "/home/kanli/Volume-Transformer-main/AMZN_2021-04-01_34200000_57600000_message_50.csv"
    # data_path = DataPath.level50[0][0]
    data_path = DataPath.level10[index][0]
    df = pd.read_csv(data_path)
    message = df[df.iloc[:,1] == 4];message.columns = ['time','type','order_id','quantity','price','side','remark']
    message = message.drop(['remark'],axis = 1)
    message = message.reset_index()
    return message

def get_orderbook_data(index):
    # data_path = "/home/kanli/Volume-Transformer-main/AMZN_2021-04-01_34200000_57600000_orderbook_50.csv";df = pd.read_csv(data_path)
    # data_path = DataPath.level50[0][1]
    data_path = DataPath.level10[index][1]
    df = pd.read_csv(data_path)
    df1 = df.iloc[:,[0,2]];df1.columns =['best_ask','best_bid']
    df1['mid_price'] = (df1.best_ask + df1.best_bid)//2
    df1 = df1.reset_index()
    return df1