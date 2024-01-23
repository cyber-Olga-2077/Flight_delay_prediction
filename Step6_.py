import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import t

data = pd.read_csv('test_data_set.csv')
data_num = data.select_dtypes(include=[np.number])
X = data_num.drop('ARR_DELAY', axis=1)
y = data_num['ARR_DELAY']
model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=50, min_samples_split=100)
model.fit(X, y)

scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

mean_rmse = np.mean(np.sqrt(-scores))

df = len(scores) - 1  #Liczba stopni swobody
alpha = 0.05  #poziom istotności
t_critical = t.ppf(1 - alpha/2, df)
std_rmse = np.std(np.sqrt(-scores))

margin_of_error = t_critical * (std_rmse / np.sqrt(len(scores)))

lower_bound = mean_rmse - margin_of_error
upper_bound = mean_rmse + margin_of_error

print('RMSE: ', mean_rmse)
print('95% Confidence Interval: ({:.4f}, {:.4f})'.format(lower_bound, upper_bound)) #przedział ufności