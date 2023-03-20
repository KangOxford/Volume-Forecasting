import pandas as pd
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
import statsmodels.api as sm


def ols(train, test):
    def out_of_sample(results, test):
        pred_x = test[0]
        pred_x = sm.add_constant(pred_x, has_constant='add')
        pred_y = results.predict(pred_x)
        real_y = test[1]
        from sklearn.metrics import r2_score
        r_squared = r2_score(real_y, pred_y)
        return r_squared

    import statsmodels.api as sm
    X = train[0]
    # X = sm.add_constant(X)
    X = sm.add_constant(X, has_constant='add')
    Y = train[1]
    results = sm.OLS(Y, X).fit()
    print(results.summary())
    return out_of_sample(results, test)

import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "//"
    data_path = path + "02_raw_component_intraSession/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    path = "/home/kanli/fifth/"
    data_path = path + "raw_component15/"
else:print("Unknown operating system")


for jump in ['open/','mid/','close/']:
    new_data_path = data_path + jump
    onlyfiles = [f for f in listdir(new_data_path) if isfile(join(new_data_path, f))]
    file = onlyfiles[0]
    def aggregate_features(file):
        dflst = pd.read_pickle(new_data_path + file)
        dflst.date = dflst.date.apply(lambda x: int(x))
        window_size = 250
        col_lst = []
        for index in tqdm(range(500)):
            print(f">>> index {index}")
            X = dflst.iloc[index:window_size+index,1:-1]; y = dflst.iloc[index:window_size+index,-1]
            def ols_with_summary(X,y):
                X = sm.add_constant(X, has_constant='add')
                results = sm.OLS(y, X).fit()
                # print(results.summary())
                return results
            model = ols_with_summary(X, y)
            def feature_selecting(model,X,y):
                selected_features = model.pvalues[1:].idxmax()
                while model.pvalues[selected_features] > 0.05:
                    X = X.drop(selected_features, axis=1)
                    if 'const' not in X.columns: X = sm.add_constant(X, has_constant='add')
                    model = sm.OLS(y, X).fit()
                    selected_features = model.pvalues[1:].idxmax()
                # print(model.summary())
                return model, X, y
            model, X, y = feature_selecting(model,X,y)
            col_lst.append(X.columns.to_list())
        return col_lst

    def filter_features(col_lst):
        import collections
        # create a flattened list of all elements in lst
        flat_lst = [item for sublist in col_lst for item in sublist]
        # count the frequency of each value in flat_lst
        freq = collections.Counter(flat_lst)
        # sort the pairs by value in ascending order
        sorted_pairs = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        # calculate the index at which to slice the list
        # cutoff = int(len(sorted_pairs) * 1.00)
        cutoff = int(len(sorted_pairs) * 0.95)
        # create a new dictionary with the remaining pairs
        filtered_dict = {k: v for k, v in sorted_pairs[:cutoff]}
        def print_results(filtered_dict):
            # print the remaining pairs in descending order
            for key, value in sorted(filtered_dict.items(), key=lambda x: x[1], reverse=True):
                print(key, value)
        # print_results(filtered_dict)
        # keys = list(filtered_dict.keys())
        # return keys
        return filtered_dict

        col_lst = aggregate_features(file)
        filtered_dict = filter_features(col_lst)
        result =pd.Series(filtered_dict)
        result.to_csv(path + jump[:-1]+ "_"+ "ols_feat_15min.csv")

