import pandas as pd
from tabulate import tabulate
import numpy as np
import plotly.express as px

#due to enormous size of our data set and the pandas' interpretetion of column of unknown type as <object> (which is quite memory-intensive) we need to manually define data type of each column

dtypes={
    "FL_DATE": "str",
    "AIRLINE": "str",
    "AIRLINE_CODE": "str",
    "FL_NUMBER": "int",
    "ORIGIN": "str",
    "ORIGIN_CITY": "str",
    "DEST": "str",
    "DEST_CITY": "str",
    "CRS_DEP_TIME": "int",
    "DEP_DELAY": "float64",
    "TAXI_OUT": "float64",
    "TAXI_IN": "float64",
    "CRS_ARR_TIME": "int",
    "ARR_DELAY": "float64",
    "CANCELLED": "bool",
    "CANCELLATION_CODE": "str",
    "DIVERTED": "bool",
    "DISTANCE": "int",
    "DELAY_DUE_CARRIER": "float64",
    "DELAY_DUE_WEATHER": "float64",
    "DELAY_DUE_NAS": "float64",
    "DELAY_DUE_SECURITY": "float64",
    "DELAY_DUE_LATE_AIRCRAFT": "float64"
}

data_set = pd.read_csv('clean_data_set.csv', dtype=dtypes)
searching_for_outliers = data_set.describe()
print(tabulate(searching_for_outliers, headers='keys', tablefmt='psql'))
searching_for_outliers_columns = data_set[["DEP_DELAY", "TAXI_OUT", "TAXI_IN", "ARR_DELAY"]]

#we find outliers using IQR method and then we cap them using 3*std method

def finding_and_capping_outliers(data_set, column_name):
    q1 = data_set[column_name].quantile(0.25)
    q3 = data_set[column_name].quantile(0.75)
    IQR = q3 - q1
    outliers = data_set[((data_set[column_name]<(q1-1.5*IQR))|(data_set[column_name]>(q3+1.5*IQR)))]
    upper_limit = data_set[column_name].mean() + 3*data_set[column_name].std()
    lower_limit = data_set[column_name].mean() - 3*data_set[column_name].std()
    data_set[column_name] = np.where(data_set[column_name] > upper_limit,
         upper_limit,
         np.where(
                data_set[column_name] < lower_limit,
                lower_limit,
                data_set[column_name])
         )
    return data_set

for column in searching_for_outliers_columns:
    data_set = finding_and_capping_outliers(data_set, column)

print(data_set.describe())