from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from Step4_transformations_and_pipeline import DISTANCE_ix
from module import calculate_dist_between_airports
import pandas as pd
import numpy as np
import joblib

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

loaded_model = joblib.load('random_forest_model.joblib')

example_data = {
    'AIRLINE': 'American Airlines Inc.',
    'AIRLINE_CODE': 'AA',
    'ORIGIN': 'CLT',
    'ORIGIN_CITY': 'Charlotte, NC',
    'DEST': 'CMH',
    'DEST_CITY': 'Columbus, OH',
    'CRS_DEP_TIME': 2040,
    'CRS_ARR_TIME': 2201,
    'DISTANCE': calculate_dist_between_airports('CLT', 'CMH'),
    'FL_DAY': 14,
    'FL_MONTH': 4,
    'FL_YEAR': 2023,
}

example_data = pd.DataFrame([example_data])

num_pipeline = joblib.load('num_pipeline.joblib')
cat_pipeline = joblib.load('categorical_pipeline.joblib')

example_data_num = example_data.select_dtypes(include=[np.number])
example_data_cat = example_data.select_dtypes(exclude=[np.number])

example_data_num_transformed = num_pipeline.transform(example_data_num)
example_data_cat_transformed = cat_pipeline.transform(example_data_cat)

num_columns = example_data_num.columns.drop('DISTANCE')

example_data_num_transformed_df = pd.DataFrame(data=example_data_num_transformed, columns=list(num_columns) + ['LOG_DISTANCE'])
example_data_cat_transformed_df = pd.DataFrame(data=example_data_cat_transformed, columns=example_data_cat.columns)

example_data = example_data_cat_transformed_df.join(example_data_num_transformed_df)

print(example_data.columns)

#Predykcje modelu
y_pred_example = loaded_model.predict(example_data)
print('Predicted Delay for the Example Data:', y_pred_example[0])