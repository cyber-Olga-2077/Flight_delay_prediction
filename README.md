# Flight_delay_prediction

Here's link to download the dirty data. [Download](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/versions/6)

Download it, extract the .zip archive, copy the file to the Flight_delay_prediction root directory.

Example in my PyCharm: [Click](https://i.imgur.com/r80fLcB.png)

First, you need to run the Start.py file to execute all the existing files. It can take a long time, so be patient.

Then you can go the the next empty Stepx_.py file and start coding. During your coding process, remember to run only the current file you are working on to avoid running the whole project again.

Remember to import the necessary libraries and functions to EVERY SINGLE ONE of the files.


Remember that you can read the data from csv files with the pandas library. You can also save the data to csv files with the pandas library. Our sets are divided into 3 files:
 
"whole_data_set" - contains all the cleaned data with deleted outliers.

"train_data_set" - contains 80% of the whole_data_set.

"test_data_set" - contains 20% of the whole_data_set.

Reading the csv file example:

```data_set = pd.read_csv('whole_data_set.csv')```

Saving the data to csv file example:

```data_set.to_csv('whole_data_set.csv', index=False, header=True)```

The most important variable that we will be predicting is the "ARR_DELAY" variable. It is the number of minutes that the flight was delayed. If the flight was delayed by 5 minutes, the "ARR_DELAY" variable will be equal to 5. If the flight was not delayed, the "ARR_DELAY" variable will be equal to 0. If the flight was advanced by 10 minutes, the "ARR_DELAY" variable will be equal to -10. And so on.

Good luck!
