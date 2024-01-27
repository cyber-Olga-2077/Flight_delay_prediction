# imports

import pandas as pd


# data reading

# Our data is divided into chunks so that we are able to read such a massive file
# on an average computing machine and save all needed data into 1 easily readable csv file.
def read_csv():
    for chunk in pd.read_csv('flights_sample_3m.csv', chunksize=3000000, dtype={"CANCELLATION_CODE": "str"}):
        yield chunk


for index, df in enumerate(read_csv()):
    df['FL_DAY'] = df['FL_DATE'].str[8:10]
    df['FL_MONTH'] = df['FL_DATE'].str[5:7]
    df['FL_YEAR'] = df['FL_DATE'].str[0:4]

    columns_to_drop = ['AIRLINE_DOT', 'DOT_CODE', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'ARR_TIME',
                       'CRS_ELAPSED_TIME', 'ELAPSED_TIME', 'CANCELLATION_CODE', 'CANCELLED', 'DIVERTED',
                       'FL_NUMBER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_SECURITY', 'DELAY_DUE_CARRIER',
                       'DELAY_DUE_NAS', 'DELAY_DUE_LATE_AIRCRAFT', 'TAXI_OUT', 'TAXI_IN', 'DEP_DELAY',
                       'AIR_TIME', 'FL_DATE']

    df.drop(columns_to_drop, inplace=True, axis=1)

    df = df.dropna(subset=["ARR_DELAY"])

    print(f"Parsed chunk {index}")

    if index == 0:
        df.to_csv('clean_data_set.csv', index=False, header=True)
    else:
        df.to_csv('clean_data_set.csv', mode='a', index=False, header=False)

