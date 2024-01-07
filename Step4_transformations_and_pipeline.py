import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from collections import Counter

data_set = pd.read_csv('train_data_set.csv') #wczytanie danych
data_set_num = data_set.select_dtypes(include=[np.number]) #wybranie wartości numerycznych
data_set_cat = data_set.select_dtypes(exclude=[np.number]) #wybranie wartości inne niż numeryczne (kategoryczne)

col_names = ["TAXI_IN", "TAXI_OUT", "AIR_TIME", "DISTANCE", "CRS_DEP_TIME"]
TAXIIN_ix, TAXIOUT_ix, AIR_TIME_ix, DISTANCE_ix, CRS_DEP_TIME_ix = [data_set_num.columns.get_loc(c) for c in col_names] #nadawanie indeksów dla kolumn

class CombinedAttrs(BaseEstimator, TransformerMixin):   #klasa transformacji
  def __init__(self, add_flight_speed=True):
    self.add_flight_speed = add_flight_speed

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    flight_speed = X[:, DISTANCE_ix] / X[:, AIR_TIME_ix]   #nowa miara
    #transformacje: log
    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_dis = log_transformer.transform(X[:, DISTANCE_ix])
    log_taxi_in = log_transformer.transform(X[:, TAXIIN_ix])
    log_taxi_out = log_transformer.transform(X[:, TAXIOUT_ix])
    #transformacje: rbf
    dep_time = data_set["CRS_DEP_TIME"]
    delays = data_set["DEP_DELAY"]

    departure_and_delay = list(zip(dep_time, delays))

    sum_delays_per_time = Counter()

    for dep_time, delay in departure_and_delay:
      sum_delays_per_time[dep_time] += delay

    #Finding the hour with the highest sum of delays:
    most_delays = max(sum_delays_per_time, key=sum_delays_per_time.get)

    print(f"The most delays occurred at hour: {most_delays}")
    rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[most_delays]], gamma=0.1))
    most_delayed_hour = rbf_transformer.transform(X[:, CRS_DEP_TIME_ix].reshape(-1, 1))

    return np.c_[X, flight_speed, log_dis, log_taxi_in, log_taxi_out, most_delayed_hour]


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

data_set_transformed.to_csv('data_set_transformed.csv', index=False, header=True)
