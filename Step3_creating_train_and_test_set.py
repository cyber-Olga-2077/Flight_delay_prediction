from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data_set = pd.read_csv('whole_data_set.csv')

data_set['ARR_DELAY_STRATA'] = pd.cut(data_set['ARR_DELAY'], bins=4, labels=[1, 2, 3, 4])
train_set, test_set = train_test_split(data_set, test_size=0.2, train_size=0.8, random_state=42, stratify=data_set['ARR_DELAY_STRATA'])

train_set.drop('ARR_DELAY_STRATA', axis=1, inplace=True)
test_set.drop('ARR_DELAY_STRATA', axis=1, inplace=True)

train_set.to_csv('train_data_set.csv', index=False, header=True)
test_set.to_csv('test_data_set.csv', index=False, header=True)
