import pandas as pd
import numpy as np
from math import sqrt
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split


data = pd.read_csv('test_set_transformed.csv')
data_num = data.select_dtypes(include=[np.number])
X = data_num.drop('ARR_DELAY', axis=1)
y = data_num['ARR_DELAY']


# Modele

# 1. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
rmse_lin = np.sqrt(-cross_val_score(lin_reg, X, y, scoring='neg_mean_squared_error', cv=10).mean())
print("Linear Regression RMSE:", rmse_lin)

# 2.Lasso Regression

param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1]
}

lasso = Lasso()
grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)
print('Lasso Best params:',grid_search.best_estimator_)
rmse_lasso = sqrt(-grid_search.best_score_)
print('Lasso Best RMSE:', rmse_lasso)

# 3. Ridge Regression

param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}


ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)
print('Ridge Best params:',grid_search.best_estimator_)
rmse_ridge = sqrt(-grid_search.best_score_)
print('Ridge Best RMSE:', rmse_ridge)

# 4. Decision Tree

param_grid = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [20, 50, 100, 200],
    'min_samples_leaf': [20, 50, 100, 200]
}


tree_reg = DecisionTreeRegressor()
grid_search = RandomizedSearchCV(tree_reg, param_grid, n_iter=16, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
grid_search.fit(X, y)
print('Tree Best params:',grid_search.best_estimator_)
best_params = grid_search.best_params_
print("Najlepsze parametry:", best_params)
rmse_tree = sqrt(-grid_search.best_score_)
print('Tree Best RMSE:', rmse_tree)

# 5. Random Forest

param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [20, 50, 100, 200],
    'min_samples_leaf': [20, 50, 100, 200],
}

forest_reg = RandomForestRegressor()
grid_search = RandomizedSearchCV(forest_reg, param_grid, n_iter=16, cv=5, scoring="neg_mean_squared_error", return_train_score=True, n_jobs=-1)
grid_search.fit(X, y)
print('Forest Best params:', grid_search.best_estimator_)
best_params = grid_search.best_params_
print("Najlepsze parametry:", best_params)
rmse_forest = sqrt(-grid_search.best_score_)
print('Forest Best RMSE:', rmse_forest)

# 6. XGBoost

param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 5, 10, 20],
    'learning_rate': [0.001, 0.01, 0.03]
}

xgb_reg = XGBRegressor()
grid_search = RandomizedSearchCV(xgb_reg, param_grid, cv=5, scoring='neg_mean_squared_error',
                                 return_train_score=True, n_jobs=-1, random_state=42)
grid_search.fit(X, y)
print('XGB params:', grid_search.best_params_)
rmse_xgb = sqrt(-grid_search.best_score_)
print('XGB Best RMSE:', rmse_xgb)



# słownik z wynikami RMSE
rmse_results = {
    'Linear Regression': rmse_lin,
    'Lasso Regression': rmse_lasso,
    'Ridge Regression': rmse_ridge,
    'Decision Tree': rmse_tree,
    'Random Forest': rmse_forest,
    'XGBoost': rmse_xgb,
}

# Znajdujemy model z najmniejszym RMSE
best_model = min(rmse_results, key=rmse_results.get)
best_rmse = rmse_results[best_model]

print(f'Best Model: {best_model}')
print(f'Best RMSE: {best_rmse}')

