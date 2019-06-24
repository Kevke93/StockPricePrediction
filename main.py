#%%
import pandas as pd 
import os 
import matplotlib as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Open the data csv as a pandas df
data_name = r"Data\DAX30.csv"
data_path = os.path.join(os.getcwd(), data_name)
df = pd.read_csv(data_path)

#Clean the Data

# Drop the First 1554 Rows because the value for opening Closing
# and High Low are the same

df = df.iloc[1554:]
df.drop("Adj Close", inplace=True, axis=1)
# Volume has a lot of 0 Values Therefore the Column will be drop
# for the beginning
df.drop("Volume", inplace=True, axis=1)
df.drop("Date", inplace=True, axis=1)
df.dropna(axis=0, inplace=True)
df.head()

#%%
# Splitt the Data Test Train, and Scale it

df_train, df_test = train_test_split(df, shuffle=False, train_size=0.8, test_size=0.2)
scaler = MinMaxScaler()

#%%
#Prepare the Datafreame for predictions
def build_time_series(df, y_col_index, time_steps):

    trainings_set = df.values

    dim_0 = df.shape[0] - time_steps
    dim_1 = df.shape[1]

    x = np.zeros((dim_0, time_steps, dim_1))
    y = np.zeros((dim_0))

    for i in range (dim_0 - 1):
        x[i] = df[i: i+ time_steps].values
        y[i] = trainings_set[i + 1, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y
#%%
times_step = 60
batch_size = 100

build_time_series(df, 3, times_step)

#%%
