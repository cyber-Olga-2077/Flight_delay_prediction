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


data = pd.read_csv('data_set_transformed.csv')
data_num = data.select_dtypes(include=[np.number])
# X = data_num.drop('ARR_DELAY', axis=1)
# y = data_num['ARR_DELAY']

random_sample = data_num.sample(n=100000)
X = random_sample.drop('ARR_DELAY', axis=1)
y = random_sample['ARR_DELAY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Modele

# 1. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
rmse_lin = np.sqrt(-cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=10).mean())
print("Linear Regression RMSE:", rmse_lin)

# 2.Lasso Regression

param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1]
}
# Lasso Best params: Lasso(alpha=0.0001)

lasso = Lasso()
grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
print('Lasso Best params:',grid_search.best_estimator_)
rmse_lasso = sqrt(-grid_search.best_score_)
print('Lasso Best RMSE:', rmse_lasso)

# 3. Ridge Regression

param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}
# Ridge Best params: Ridge(alpha=0.01, solver='saga')

ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
print('Ridge Best params:',grid_search.best_estimator_)
rmse_ridge = sqrt(-grid_search.best_score_)
print('Ridge Best RMSE:', rmse_ridge)

# 4. Decision Tree

param_grid = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [20, 50, 100, 200],
    'min_samples_leaf': [20, 50, 100, 200]
}
# Tree Best params: DecisionTreeRegressor(max_depth=20, min_samples_leaf=50, min_samples_split=100)

tree_reg = DecisionTreeRegressor()
grid_search = RandomizedSearchCV(tree_reg, param_grid, n_iter=16, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
grid_search.fit(X_train, y_train)
print('Tree Best params:',grid_search.best_estimator_)
rmse_tree = sqrt(-grid_search.best_score_)
print('Tree Best RMSE:', rmse_tree)

# 5. Random Forest

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [20, 50, 100, 200],
    'min_samples_leaf': [20, 50, 100, 200],
}
# Forest Best params: RandomForestRegressor(max_depth=20, min_samples_leaf=50, min_samples_split=100)

forest_reg = RandomForestRegressor()
grid_search = RandomizedSearchCV(forest_reg, param_grid, n_iter=10, cv=5, scoring="neg_mean_squared_error", return_train_score=True, n_jobs=-1)
grid_search.fit(X_train, y_train)
print('Forest Best params:', grid_search.best_estimator_)
rmse_forest = sqrt(-grid_search.best_score_)
print('Forest Best RMSE:', rmse_forest)

# 6. XGBoost

param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 5, 10, 20],
    'learning_rate': [0.001, 0.01, 0.03]
}
# XGB params: {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.03}

xgb_reg = XGBRegressor()
grid_search = RandomizedSearchCV(xgb_reg, param_grid, cv=5, scoring='neg_mean_squared_error',
                                 return_train_score=True, n_jobs=-1, random_state=42)
grid_search.fit(X_train, y_train)
print('XGB params:', grid_search.best_params_)
rmse_xgb = sqrt(-grid_search.best_score_)
print('XGB Best RMSE:', rmse_xgb)


# 7. SVM

param_grid = {
    'C': [1, 10, 100],
    'epsilon': [0.01, 0.1, 0.2],
    'kernel': ['rbf', 'linear']
}

svr = SVR()
grid_search = RandomizedSearchCV(svr, param_grid, n_iter=6, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

print('SVR params:', grid_search.best_estimator_)
rmse_svr = sqrt(-grid_search.best_score_)
print('SVR Best RMSE:', rmse_svr)

# 8. K-nearest Neighbors

param_grid = {
    'n_neighbors': [5, 10, 15, 50],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn_reg = KNeighborsRegressor()
grid_search = RandomizedSearchCV(knn_reg, param_grid,
                                 n_iter=8, cv=5, scoring='neg_mean_squared_error',
                                 return_train_score=True, n_jobs=-1, random_state=42)
grid_search.fit(X_train, y_train)
print('KNN params:', grid_search.best_estimator_)
rmse_knn = sqrt(-grid_search.best_score_)
print('KNN Best RMSE:', rmse_knn)


# Tworzymy słownik z wynikami RMSE
rmse_results = {
    'Linear Regression': rmse_lin,
    'Lasso Regression': rmse_lasso,
    'Ridge Regression': rmse_ridge,
    'Decision Tree': rmse_tree,
    'Random Forest': rmse_forest,
    'XGBoost': rmse_xgb,
    'SVR': rmse_svr,
    'K-nearest Neighbors': rmse_knn
}

# Znajdujemy model z najmniejszym RMSE
best_model = min(rmse_results, key=rmse_results.get)
best_rmse = rmse_results[best_model]

print(f'Best Model: {best_model}')
print(f'Best RMSE: {best_rmse}')

