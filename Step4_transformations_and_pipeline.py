import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from collections import Counter

from tabulate import tabulate

train_set = pd.read_csv('train_data_set.csv') #wczytanie danych

train_set_Y = train_set['ARR_DELAY']
train_set_X = train_set.drop('ARR_DELAY', axis=1)

train_set_num = train_set_X.select_dtypes(include=[np.number]) #wybranie wartości numerycznych
train_set_cat = train_set_X.select_dtypes(exclude=[np.number]) #wybranie wartości inne niż numeryczne (kategoryczne)

column_names = train_set_num.columns.drop('DISTANCE')

DISTANCE_ix = train_set_num.columns.get_loc("DISTANCE") #pobranie indeksu kolumny DISTANCE


class CombinedAttrs(BaseEstimator, TransformerMixin):   #klasa transformacji
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
        LOG_DISTANCE = log_transformer.transform(X[:, DISTANCE_ix])
        X = np.delete(X, DISTANCE_ix, axis=1)

        return np.c_[X, LOG_DISTANCE]


categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('label_encoder', OrdinalEncoder()),
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('adder', CombinedAttrs()),
    ('std_scaler', StandardScaler())
])

# łączenie przetransformowanych wartości numerycznych z kategorycznymi w formacie surowym
train_set_transformed_num = num_pipeline.fit_transform(train_set_num)
train_set_transformed_num_df = pd.DataFrame(data=train_set_transformed_num, columns=list(column_names) + ['LOG_DISTANCE'])
train_set_transformed_cat = categorical_pipeline.fit_transform(train_set_cat)
train_set_transformed_cat_df = pd.DataFrame(data=train_set_transformed_cat, columns=train_set_cat.columns)
train_set_transformed = train_set_transformed_cat_df.join(train_set_transformed_num_df).join(train_set_Y)
train_set_transformed.to_csv('train_set_transformed.csv', index=False, header=True)

test_set = pd.read_csv('test_data_set.csv') # wczytanie danych
test_set_Y = test_set['ARR_DELAY']
test_set_X = test_set.drop('ARR_DELAY', axis=1)

test_set_num = test_set_X.select_dtypes(include=[np.number]) # wybranie wartości numerycznych
test_set_cat = test_set_X.select_dtypes(exclude=[np.number]) # wybranie wartości inne niż numeryczne (kategoryczne)

# łączenie przetransformowanych wartości numerycznych z kategorycznymi w formacie surowym
test_set_transformed_num = num_pipeline.transform(test_set_num)
test_set_transformed_num_df = pd.DataFrame(data=test_set_transformed_num, columns=list(column_names) + ['LOG_DISTANCE'])
test_set_transformed_cat = categorical_pipeline.transform(test_set_cat)
test_set_transformed_cat_df = pd.DataFrame(data=test_set_transformed_cat, columns=test_set_cat.columns)
test_set_transformed = test_set_transformed_cat_df.join(test_set_transformed_num_df).join(test_set_Y)
test_set_transformed.to_csv('test_set_transformed.csv', index=False, header=True)

joblib.dump(num_pipeline, 'num_pipeline.joblib')
joblib.dump(categorical_pipeline, 'categorical_pipeline.joblib')