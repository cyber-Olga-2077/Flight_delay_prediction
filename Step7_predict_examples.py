from module import calculate_dist_between_airports
import pandas as pd
import numpy as np
import joblib

loaded_model = joblib.load('random_forest_model.joblib')

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

num_rows_to_read = 100
train_set_partial = pd.read_csv('train_data_set.csv', nrows=num_rows_to_read)
train_set_num = train_set_partial.select_dtypes(include=[np.number])
X_train = train_set_num.drop('ARR_DELAY', axis=1)

#Dodanie brakujacych w przykladzie kolumn z modelu
required_features = set(X_train.columns)
for col in required_features:
    if col not in example_data:
        example_data[col] = np.nan

#Zmiana kolejnosci kolumn na zgodna z modelem
example_data = example_data[X_train.columns]

#Predykcje modelu
y_pred_example = loaded_model.predict(example_data)
print('Predicted Delay for the Example Data:', y_pred_example[0])