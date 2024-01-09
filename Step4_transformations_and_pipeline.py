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

data_set = pd.read_csv('train_data_set.csv.csv') #użyty ten plik zamiast whole_data_set,ponieważ były w nim potrzebne mi kolumny (więcej na messengerze ;), ale bedzie można zmienić na whole_data_set )
data_set_num = data_set.select_dtypes(include=[np.number]) #wybranie wartości numerycznych
data_set_cat = data_set.select_dtypes(exclude=[np.number]) #wybranie wartości inne niż numeryczne (kategoryczne)

col_names = ["TAXI_IN", "TAXI_OUT", "AIR_TIME", "DISTANCE", "CRS_DEP_TIME"]
TI_ix, TO_ix, AT_ix, D_ix, CDT_ix = [data_set_num.columns.get_loc(c) for c in col_names] #nadawanie indeksów dla kolumn

class CombinedAttrs(BaseEstimator, TransformerMixin):   #klasa transformacji
  def __init__(self, add_flight_speed=True):
    self.add_flight_speed = add_flight_speed

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    flight_speed = X[:,D_ix] / X[:,AT_ix]   #nowa miara
    #transformacje: log
    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_dis = log_transformer.transform(X[:,D_ix])
    log_taxi_in = log_transformer.transform(X[:,TI_ix])
    log_taxi_out = log_transformer.transform(X[:,TO_ix])
    #transformacje: rbf
    dep_time = data_set["DEP_TIME"]
    delays = data_set["DEP_DELAY"]

    departure_and_delay = list(zip(dep_time, delays))

    sum_delays_per_time = Counter()

    for dep_time, delay in departure_and_delay:
      sum_delays_per_time[dep_time] += delay

    #Finding the hour with the highest sum of delays:
    most_delays = max(sum_delays_per_time, key=sum_delays_per_time.get)

    print(f"The most delays occurred at hour: {most_delays}")
    rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[most_delays]], gamma=0.1))
    delay_time_1800 = rbf_transformer.transform(X[:,CDT_ix].reshape(-1, 1))

    return np.c_[X, flight_speed, log_dis, log_taxi_in, log_taxi_out, delay_time_1800]


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
