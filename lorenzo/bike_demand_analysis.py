import pandas as pd
import numpy as np
import requests
import zipfile
import io
import copy
from sklearn.ensemble import AdaBoostRegressor
#import ahead

content = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip").content
with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(arc.open("day.csv"), header=0, sep=',', parse_dates=['dteday'], index_col='dteday')

da = raw_data[(raw_data.index.year == 2011) & ((3 < raw_data.index.month) & (raw_data.index.month < 10))]
db = raw_data[(raw_data.index.year == 2012) & ((3 < raw_data.index.month) & (raw_data.index.month < 10))]
dc = raw_data[(raw_data.index.year == 2011) & (( 3 >= raw_data.index.month) |(raw_data.index.month >= 10))]
dd = raw_data[(raw_data.index.year == 2011) & (( 3 >= raw_data.index.month) | (raw_data.index.month >= 10))]

d1 =  copy.deepcopy(pd.concat([da, db])) # winter
d2 = copy.deepcopy(pd.concat([dc,dd])) # summer

train = copy.deepcopy(raw_data[(raw_data.index.day<20) & ((raw_data.index.year == 2011) | ((raw_data.index.year == 2012) &(raw_data.index.month < 8)))])
val = copy.deepcopy(raw_data[(raw_data.index.day>=20) & ((raw_data.index.year == 2011) | ((raw_data.index.year == 2012) &(raw_data.index.month < 8)))])
test = copy.deepcopy(raw_data[(raw_data.index.year == 2012) &(raw_data.index.month >= 8)])

X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]
X_val = val.iloc[:,:-1]
y_val = val.iloc[:,-1]

model = AdaBoostRegressor()
model.fit(X_train,y_train)
res = model.predict(X_val)
MAPE = np.mean(np.abs(y_val - res) / y_val) * 100
