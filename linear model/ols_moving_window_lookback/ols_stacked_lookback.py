import numpy as np
import pandas as pd
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
import warnings;warnings.simplefilter("ignore", category=FutureWarning)
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score

def out_of_sample(results, test):
    pred_x = test[0]
    pred_x = sm.add_constant(pred_x, has_constant='add')
    pred_y = results.predict(pred_x)
    real_y = test[1]
    r_squared = r2_score(real_y, pred_y)
    return r_squared

def ols(train, test):
    X = train[0]
    X = sm.add_constant(X, has_constant='add')
    Y = train[1]
    results = sm.OLS(Y, X).fit()
    print(results.summary())
    return results, out_of_sample(results, test)

path = "/home/kanli/forth/"
# data_path = path + "out_overlap5/"
data_path = path + "out_overlap15/"
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
# for i in tqdm(range(len(onlyfiles))):
file = onlyfiles[0]
df = pd.read_pickle(data_path + file)


X = df.iloc[:,1:-1]
y = df.iloc[:,-1]

'''next'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model, out_of_sample = ols((X_train, y_train), (X_test, y_test))
print(f">>> out_of_sample: {out_of_sample}")
selected_features = model.pvalues[1:].idxmax()
while model.pvalues[selected_features] > 0.05:
    X = X.drop(selected_features, axis=1)
    model = sm.OLS(y, X).fit()
    selected_features = model.pvalues[1:].idxmax()
print(model.summary())

X_test = X_test[X.columns]
real_y = y_test
pred_y = model.predict(X_test)
r_squared = r2_score(real_y, pred_y)
print(f">>> out_of_sample: {r_squared}")



'''next'''
oos_lst = []
columns_list = []
for state in range(1000):
    # state = 1 #$
    print(f">>>>> state {state}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=state)
    X_train = sm.add_constant(X_train, has_constant='add')
    X_test = sm.add_constant(X_test, has_constant='add')
    model = sm.OLS(y_train, X_train).fit()
    # print(model.summary())

    selected_features = model.pvalues[1:].idxmax()
    while model.pvalues[selected_features] > 0.05:
        X_train = X_train.drop(selected_features, axis=1)
        model = sm.OLS(y_train, X_train).fit()
        selected_features = model.pvalues[1:].idxmax()
    # print(model.summary())

    X_test = X_test[X_train.columns]
    # X_test = sm.add_constant(X_test, has_constant='add')
    real_y = y_test
    pred_y = model.predict(X_test)
    r_squared = r2_score(real_y, pred_y)
    print(f">>> out_of_sample new: {r_squared}")
    oos_lst.append(r_squared)
    columns_list.append(X_train.columns)
np.mean(oos_lst)

window_size = 30
kernel = np.ones(window_size) / window_size
moving_avg = np.convolve(oos_lst, kernel, mode='valid')

import matplotlib.pyplot as plt; plt.rcParams["figure.figsize"] = (40,20)
plt.plot(oos_lst[window_size:],label='OUT OF SAMPLE R_SQUARED')
plt.plot(moving_avg,label='AVERAGED_OUT OF SAMPLE R_SQUARED', linewidth=4)
plt.xticks(fontsize=20);plt.yticks(fontsize=20);
plt.legend(fontsize=20);
plt.show()

columns_df  = pd.DataFrame(columns_list)


