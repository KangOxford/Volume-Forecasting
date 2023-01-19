# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd 



class DataPath:
    # level50 = [
    #     ("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_50.csv",
    #      "/Users/kang/Desktop/Volume-Tranformer/AMZN_2021-04-01_34200000_57600000_orderbook_50.csv"),
    #     ]
    # level10 = [
    #     ("/Users/kang/Data/AMZN_2021/AMZN_2021-04-26_34200000_57600000_message_10.csv",
    #      "/Users/kang/Data/AMZN_2021/AMZN_2021-04-26_34200000_57600000_orderbook_10.csv"),
    #
    #     ("/Users/kang/Data/AMZN_2021/AMZN_2021-04-21_34200000_57600000_message_10.csv",
    #       "/Users/kang/Data/AMZN_2021/AMZN_2021-04-21_34200000_57600000_orderbook_10.csv"),
    #
    #     ("/Users/kang/Data/AMZN_2021/AMZN_2021-04-20_34200000_57600000_message_10.csv",
    #       "/Users/kang/Data/AMZN_2021/AMZN_2021-04-20_34200000_57600000_orderbook_10.csv"),
    #
    #     ("/Users/kang/Data/AMZN_2021/AMZN_2021-04-19_34200000_57600000_message_10.csv",
    #       "/Users/kang/Data/AMZN_2021/AMZN_2021-04-19_34200000_57600000_orderbook_10.csv"),
    #
    #     ("/Users/kang/Data/AMZN_2021/AMZN_2021-04-15_34200000_57600000_message_10.csv",
    #       "/Users/kang/Data/AMZN_2021/AMZN_2021-04-15_34200000_57600000_orderbook_10.csv")
    #     ]
    def __init__(self):
        self.file_dir = "/Users/kang/Data/AMZN_2021/"
        name_list = [file for file in os.listdir(self.file_dir) if file[-3:] == 'csv']
        level10_list = [name for name in name_list if name[-6:-4] == '10']
        level50_list = [name for name in name_list if name[-6:-4] == '50']
        def process_list(lst):
            # lst = level10_list
            lst = sorted(lst)
            rst = [(lst[2*i], lst[2*i+1]) for i in range(len(lst)//2)]
            return rst
        self.level10 = process_list(level10_list)
        self.level50 = process_list(level50_list)


    def get_orderbook_data(self,index):
        data_path = self.level10[index][1]
        df = pd.read_csv(self.file_dir + data_path)
        df1 = df.iloc[:,[0,2]];df1.columns =['best_ask','best_bid']
        df1['mid_price'] = (df1.best_ask + df1.best_bid)//2
        df1 = df1.reset_index()
        return df1

    def get_message_data(self,index):
        data_path = self.level10[index][0]
        df = pd.read_csv(self.file_dir + data_path)
        message = df[df.iloc[:,1] == 4];message.columns = ['time','type','order_id','quantity','price','side','remark']
        message = message.drop(['remark'],axis = 1)
        message = message.reset_index()
        return message

def get_message_data(index):
    # data_path = "/home/kanli/Volume-Transformer-main/AMZN_2021-04-01_34200000_57600000_message_50.csv"
    # data_path = DataPath.level50[0][0]
    datapath = DataPath()
    data_path = datapath.level10[index][0]
    df = pd.read_csv(datapath.file_dir + data_path)
    message = df[df.iloc[:,1] == 4];message.columns = ['time','type','order_id','quantity','price','side','remark']
    message = message.drop(['remark'],axis = 1)
    message = message.reset_index()
    return message

def get_orderbook_data(index):
    # data_path = "/home/kanli/Volume-Transformer-main/AMZN_2021-04-01_34200000_57600000_orderbook_50.csv";df = pd.read_csv(data_path)
    # data_path = DataPath.level50[0][1]
    datapath = DataPath()
    data_path = datapath.level10[index][1]
    df = pd.read_csv(datapath.file_dir + data_path)
    df1 = df.iloc[:,[0,2]];df1.columns =['best_ask','best_bid']
    df1['mid_price'] = (df1.best_ask + df1.best_bid)//2
    df1 = df1.reset_index()
    return df1

if __name__ == "__main__":
    ob = get_orderbook_data(index = 0)
