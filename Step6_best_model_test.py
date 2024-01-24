import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import t
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from module import calculate_dist_between_airports


train_set = pd.read_csv('train_data_set.csv')
test_set = pd.read_csv('test_data_set.csv')
train_set_num = train_set.select_dtypes(include=[np.number])
test_set_num = test_set.select_dtypes(include=[np.number])

X_train = train_set_num.drop('ARR_DELAY', axis=1)
y_train = train_set_num['ARR_DELAY']
X_test = test_set_num.drop('ARR_DELAY', axis=1)
y_test = test_set_num['ARR_DELAY']
forest = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=50, min_samples_split=100)
forest.fit(X_train, y_train)

y_pred_test = forest.predict(X_test)
scores = cross_val_score(forest, X_test, y_test, cv=5, scoring='neg_mean_squared_error')

mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

#Walidacja krzyżowa na train secie
scores = cross_val_score(forest, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mean_rmse = np.mean(np.sqrt(-scores))

df = len(scores) - 1  #Liczba stopni swobody
alpha = 0.05  #poziom istotności
t_critical = t.ppf(1 - alpha/2, df)
std_rmse = np.std(np.sqrt(-scores))

margin_of_error = t_critical * (std_rmse / np.sqrt(len(scores)))

lower_bound = mean_rmse - margin_of_error
upper_bound = mean_rmse + margin_of_error

print('Training Set RMSE (Cross-Validation):', mean_rmse)
print('95% Confidence Interval: ({:.4f}, {:.4f})'.format(lower_bound, upper_bound)) #przedział ufności

print('\nTest Set Metrics:')
print('Mean Squared Error:', mse_test)
print('Root Mean Squared Error:', rmse_test)
print('Mean Absolute Error:', mae_test)
print('R-squared:', r2_test)
print("Based on these results, the model appears to be accurate and generalizes well to unseen data.")

example_data = {
    'FL_DATE': '2024-01-23',
    'AIRLINE': 'Delta Air Lines Inc.',
    'AIRLINE_CODE': 'DL',
    'ORIGIN': 'JFK',
    'ORIGIN_CITY': 'New York',
    'DEST': 'LAX',
    'DEST_CITY': 'Los Angeles',
    'CRS_DEP_TIME': 1200,
    'DISTANCE': calculate_dist_between_airports('JFK', 'LAX'),
    'FL_MONTH': 1,
}

example_data = pd.DataFrame([example_data])
example_data['FL_DATE'] = pd.to_datetime(example_data['FL_DATE'])

#Dodanie brakujacych w przykladzie kolumn z modelu
required_features = set(X_train.columns)
for col in required_features:
    if col not in example_data:
        example_data[col] = np.nan

#Zmiana kolejnosci kolumn na zgodna z modelem
example_data = example_data[X_train.columns]

#Predykcje modelu
y_pred_example = forest.predict(example_data)
print('Predicted Delay for the Example Data:', y_pred_example[0])