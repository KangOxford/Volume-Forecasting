# -*- coding: utf-8 -*-
import pandas as pd
from data_processing.data_pipeline import get_data

def get_feature_label(data):
    
    columns = data.columns.tolist()
    # columns.remove("vol_direction")
    # columns.remove("vol_change")
    features =  data[columns]
    labels = data.vol_direction
    features.timeHM_start = features.timeHM_start.apply(lambda x: int(x[0:2]) + int(x[2:-1])*0.01)
    features.timeHM_end = features.timeHM_end.apply(lambda x: int(x[0:2]) + int(x[2:-1])*0.01)
    return features, labels


def select_features(features):
    columns = features.columns
    # columns = ['timeHM_start','timeHM_end',"ask_num_orders",'ask_notional',"bid_num_orders",'bid_notional','volume']
    # columns = ['timeHM_start','timeHM_end',"ask_num_orders",'ask_notional']
    # columns = ['intrady_session', 'bid_num_orders', 'ask_num_orders', 'bid_volume', 'ask_volume', 'bid_notional', 'ask_notional', 'ag_bid_num_orders', 'ag_ask_num_orders']
    # columns = ['intrady_session', 'bid_num_orders', 'ask_num_orders', 'bid_volume', 'ask_volume', 'bid_notional', 'ask_notional']
    result = features[columns]
    return result 

def select_lables(data):
    return data.volume




if __name__ == "__main__":
    
    data = get_data()
    data.dropna(inplace=True)
    
    # ----------- 01 -----------
    # # given intrady_session
    # data = data[data.intrady_session == 1]
    # data = data.drop(['intrady_session'], axis = 1)
    
    
    # ----------- 02 -----------
    features, labels = get_feature_label(data)
    labels = select_lables(data)
    features = select_features(features)
    
    
    labels = labels.shift(-1).dropna()
    features = features.iloc[:-1,:]
    
    # ----------- 03 -----------
    size = 10
    import statsmodels.api as sm
    X = features[:-size]
    X = sm.add_constant(X)
    Y = pd.DataFrame(labels[:-size])
    results = sm.OLS(Y,X).fit()

    print(results.summary())
    
    # ----------- 04 -----------
    y = results.predict(X)
    pred_x = features[-size:]
    pred_x = sm.add_constant(pred_x, has_constant='add')
    pred_y = results.predict(pred_x)
    real_y = labels[-size:]
    
    
    from sklearn.metrics import mean_squared_error, r2_score
    r_squared = r2_score(real_y,pred_y)
