import xgboost as xgb
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline
import pandas as pd
from os import listdir;from os.path import isfile, join
import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    path = "/home/kanli/seventh/"
else:print("Unknown operating system")
data_path = path + "02_raw_component/"
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
# import warnings
# warnings.filterwarnings("ignore")
if __name__=="__main__":
    for i in tqdm(range(len(onlyfiles))): # on mac4
        bin_size = 26
        num_day = 10
        file = onlyfiles[i]
        # if file in already_done:
        #     print(f"++++ {j}th {file} is already done before")
        #     continue
        print(f">>>> {i}th {file}")
        dflst = pd.read_pickle(data_path + file)
        counted = dflst.groupby("date").count()
        date = pd.DataFrame(counted[counted['VO'] ==26].index)
        dflst = pd.merge(dflst, date, on = "date")
        assert dflst.shape[0]/ bin_size ==  dflst.shape[0]// bin_size

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(dflst.iloc[:,4:-1])
        dflst.iloc[:,4:-1] = scaler.transform(dflst.iloc[:,4:-1])
        times = np.arange(0,dflst.shape[0]//bin_size-num_day)
        for time in times:
            index = bin_size * time
            train_x = dflst.iloc[index:index+bin_size*num_day,4:-1]
            train_y = dflst.iloc[index:index+bin_size*num_day:,-1]
            valid_x = dflst.iloc[index+bin_size*num_day:index+bin_size*(num_day+1),4:-1]
            valid_y = dflst.iloc[index+bin_size*num_day:index+bin_size*(num_day+1),-1].values
            dtrain = xgb.DMatrix(train_x, train_y)
            parameters = {
                          'max_depth': [5, 10, 15, 20, 25],
                          'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
                          'n_estimators': [500, 1000, 2000, 3000, 5000],
                          'min_child_weight': [0, 2, 5, 10, 20],
                          'max_delta_step': [0, 0.2, 0.6, 1, 2],
                          'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
                          'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
                          'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
                          'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
                          'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
            }
            num_rounds = 500
            plst = parameters.items()
            model = xgb.train(plst, dtrain, num_rounds)

            # 对测试集进行预测
            dtest = xgb.DMatrix(X_test)
            ans = model.predict(dtest)
            # 对测试集进行预测
            dtest = xgb.DMatrix(X_test)
            ans = model.predict(dtest)

            # 显示重要特征
            plot_importance(model)
            plt.show()

            xlf = xgb.XGBClassifier(max_depth=10,
                        learning_rate=0.01,
                        n_estimators=2000,
                        silent=True,
                        objective='multi:softmax',
                        num_class=3 ,
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=0,
                        missing=None)
            print("start search")
            gs = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)
            print("end search, gs: ",gs)
            gs.fit(train_x, train_y)
            print("end fit")

            print("Best score: %0.3f" % gs.best_score_)
            print("Best parameters set: %s" % gs.best_params_ )
            break
        break
