# -*- coding: utf-8 -*-

from data_processing.data_pipeline import get_data

def get_feature_label(data):
    data = data.copy()
    data.dropna(inplace=True)
    columns = data.columns.tolist()
    columns.remove("vol_direction")
    columns.remove("vol_change")
    features =  data[columns]
    labels = data.vol_direction
    features.timeHM_start = features.timeHM_start.apply(lambda x: int(x[0:2]) + int(x[2:-1])*0.01)
    features.timeHM_end = features.timeHM_end.apply(lambda x: int(x[0:2]) + int(x[2:-1])*0.01)
    return features, labels


def select_features(features):
    columns = ['timeHM_start','timeHM_end',"ask_num_orders",'ask_notional',"bid_num_orders",'bid_notional','volume']
    result = features[columns]
    return result 

if __name__ == "__main__":
    data = get_data()
    features, labels = get_feature_label(data)
    features = select_features(features)
    
    
    import statsmodels.api as sm
    X = features[:-10]
    X = sm.add_constant(X)
    Y = labels[:-10]
    model = sm.OLS(Y,X)
    results = model.fit()

    # results.params
    # results.tvalues
    results.summary()
