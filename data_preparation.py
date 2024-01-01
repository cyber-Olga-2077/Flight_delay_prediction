#imports

import pandas as pd

#data reading
#Our data is divided into chunks so that we are able to read such a massive file on an average computing machine and save all needed data into 1 easily readable csv file.
def read_csv():
    for chunk in pd.read_csv('_ALL_FLIGHTS_30m.csv', chunksize=5000000, dtype={
        "CANCELLATION_CODE": "str",
    }):
       yield chunk

for index, df in enumerate(read_csv()):
    df.drop('AIRLINE_DOT', inplace=True, axis=1)
    df.drop('DOT_CODE', inplace=True, axis=1)
    df.drop('DEP_TIME', inplace=True, axis=1)
    df.drop('WHEELS_OFF', inplace=True, axis=1)
    df.drop('WHEELS_ON', inplace=True, axis=1)
    df.drop('ARR_TIME', inplace=True, axis=1)
    df.drop('CRS_ELAPSED_TIME', inplace=True, axis=1)
    df.drop('ELAPSED_TIME', inplace=True, axis=1)
    df.drop('AIR_TIME', inplace=True, axis=1)
    #print(tabulate(df, headers='keys', tablefmt='psql'))
    histogram = df.hist(bins=50, figsize=(12, 10))
    figure = histogram[0][0].get_figure()
    figure.savefig(f'wykres.pdf')
    print(f"Parsed chunk {index}")
    if index == 0:
        df.to_csv('clean_data_set.csv', index=False, header=True)
    else:
        df.to_csv('clean_data_set.csv', mode='a', index=False, header=False)