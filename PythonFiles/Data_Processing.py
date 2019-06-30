# This Script provides all function for Data Clearning
import pandas as pd 
import numpy as np

def inital_clearing(DATA_PATH, FEATURES):

    # read in the Data
    df = pd.read_csv(DATA_PATH)
    # remove the unncecessary collumns
    df = df.loc[:, FEATURES ]

    # Drop all the rows which include a Null value
    df.dropna(axis=0, inplace=True)
    
    # return the df
    return df


def build_time_series_LSTM(data_arr, y_col_index, time_steps):
    # A Timeseries is Build, inwhich the Data_arr is splitt in x and y
    # x is a 3 dim Arr. x[0] == Data, x[1]= Time_Steps, x[2] = Amount of features
    dim_0 = data_arr.shape[0] - time_steps
    dim_1 = data_arr.shape[1]

    x = np.zeros(( dim_0, time_steps, dim_1))
    y = np.zeros((dim_0))

    for i in range(dim_0 - 1):
        x[i] = data_arr[i: i + time_steps]
        y[i] = data_arr[i + 1, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y



def trim_data_set(data_arr, batch_size):
    # the DataSet has to be dividible through the Batch size
    rows_drop = data_arr.shape[0] % batch_size

    if rows_drop > 0:
        return data_arr[:- rows_drop ]
    else:
        return data_arr


def build_data_LSTM(df_train, df_test, y_col_index, time_steps, batch_size):
    # Build up the Data for creating a LSTM Model

    x_train, y_train = build_time_series_LSTM(df_train, y_col_index, time_steps)
    x_train = trim_data_set(x_train, batch_size)
    y_train = trim_data_set(y_train, batch_size)

    x_test, y_test = build_time_series_LSTM(df_test, y_col_index, time_steps)
    x_test = trim_data_set(x_test, batch_size)
    y_test = trim_data_set(y_test, batch_size)

    return x_train, y_train, x_test, y_test
