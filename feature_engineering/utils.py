# -*- coding: utf-8 -*-

def data_split(data,size = 10):
    X = data.iloc[:-size,:-1]
    Y = data.iloc[:-size,-1]
    pred_x = data.iloc[-size:,:-1]
    real_y = data.iloc[-size:,-1]  
    return (X,Y),(pred_x,real_y)