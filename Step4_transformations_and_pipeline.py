import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer

data_set = pd.read_csv('flights_sample_3m.csv') #użyty ten plik zamiast whole_data_set,ponieważ były w nim potrzebne mi kolumny (więcej na messengerze ;), ale bedzie można zmienić na whole_data_set )
data_set_num = data_set.select_dtypes(include=[np.number]) #wybranie wartości numerycznych
data_set_cat = data_set.select_dtypes(exclude=[np.number]) #wybranie wartości inne niż numeryczne (kategoryczne)

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

col_names = ["TAXI_IN", "TAXI_OUT", "AIR_TIME", "DISTANCE", "CRS_DEP_TIME"]
r_ix, b_ix, p_ix, h_ix, j_ix = [data_set_num.columns.get_loc(c) for c in col_names] #nadawanie indeksów dla kolumn

class CombinedAttrs(BaseEstimator, TransformerMixin):   #klasa transformacji
  def __init__(self, add_flight_speed=True):
    self.add_flight_speed = add_flight_speed

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    flight_speed = X[:,h_ix] / X[:,p_ix]   #nowa miara
    #transformacje: log
    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_dis = log_transformer.transform(X[:,h_ix])
    log_taxi_in = log_transformer.transform(X[:,r_ix])
    log_taxi_out = log_transformer.transform(X[:,b_ix])
    #transformacje: rbf
    rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[1800]], gamma=0.1)) # 1800 -> statystycznie najwięcej opóźnień ma miejsce o godzinie 18:00
    delay_time_1800 = rbf_transformer.transform(X[:,j_ix].reshape(-1, 1))

    return np.c_[X, flight_speed, log_dis, log_taxi_in, log_taxi_out, delay_time_1800]


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('adder', CombinedAttrs()),
    ('std_scaler', StandardScaler())
])
 #łączenie przetransformowanych wartości numerycznych z kategorycznymi w formacie surowym
data_set_transformed_num = num_pipeline.fit_transform(data_set_num)
data_set_transformed_num_df = pd.DataFrame(data=data_set_transformed_num, columns=list(data_set_num.columns) + ['flight_speed', 'log_dis', 'log_taxi_in', 'log_taxi_out', 'delay_time_1800'])
data_set_transformed = data_set_cat.join(data_set_transformed_num_df)
#print(data_set_transformed)
