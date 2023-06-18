import pandas as pd


from config import *


# r2df = pd.read_csv(path00 +"07_r2df.csv", index_col=0)
# r2df = pd.read_csv(path00 +"07_r2df_Lasso_.csv", index_col=0)
r2df = pd.read_csv(path00 +"07_r2df_Ridge_.csv", index_col=0)
# r2df = pd.read_csv(path00 +"07_r2df.csv", index_col=0)
r2df['mean'] = r2df.mean(axis=1)
mean_row = pd.DataFrame(r2df.mean(axis=0), columns=['mean']).T
r2df = pd.concat([r2df, mean_row])
# r2df['mean'].mean()
mean_col = r2df['mean'][:-1]
mean_row = mean_row.iloc[:,:-1]
percentage = 1.00
# percentage = 0.95
# percentage = 0.90
# percentage = 0.85
# percentage = 0.78
# percentage = 0.75
# percentage = 0.10
# percentage = 0.05
top_95_percent_col = mean_col[mean_col >= mean_col.quantile(1 - percentage)]
date_mean = top_95_percent_col.mean()
top_95_percent_row = mean_row.apply(lambda row: row[row >= row.quantile(1 - percentage)], axis=1)
asset_mean = top_95_percent_row.mean(axis = 1)
print(date_mean, asset_mean)


peakAssetsRemovedColumns = top_95_percent_row.columns
nr2df = r2df.loc[:,peakAssetsRemovedColumns]
nr2df['mean'] = nr2df.mean(axis=1)
nr2df['mean'].mean()
