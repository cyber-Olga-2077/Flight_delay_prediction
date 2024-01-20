import pandas as pd
import numpy as np
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

data = pd.read_csv('data_set_transformed.csv')
data_num = data.select_dtypes(include=[np.number])
X = data_num.drop('ARR_DELAY', axis=1)
y = data_num['ARR_DELAY']

model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=50, min_samples_split=100)
model.fit(X, y)

scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

mean_rmse = np.mean(np.sqrt(-scores))
print('RMSE: ', mean_rmse)



