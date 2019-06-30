# %%
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from sklearn.metrics import r2_score
from keras.models import load_model

# Open the data csv as a pandas df

output_path = r"C:\Users\admin\Desktop\Masterarbeit\Programm\StockPricePrediction\Output"
data_name = r"Data\DAX30.csv"
data_path = os.path.join(os.getcwd(), data_name)
print(data_path)
df = pd.read_csv(data_path)

# Clean the Data

# Drop the First 1554 Rows because the value for opening Closing
# and High Low are the same


df.drop("Adj Close", inplace=True, axis=1)

# Volume has a lot of 0 Values Therefore the Column will be drop
# for the beginning
df.drop("Volume", inplace=True, axis=1)
df.drop("Date", inplace=True, axis=1)
df.dropna(axis=0, inplace=True)
df.info()

# %%
# Splitt the Data Test Train, and Scale it

df_train, df_test = train_test_split(df, shuffle=False, train_size=0.8,\
     test_size=0.2)

scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(df_train)
test = scaler.fit_transform(df_test)

#%%
#Prepare the Datafreame for predictions
def build_time_series(data_arr, y_col_index, time_steps):

    dim_0 = data_arr.shape[0] - time_steps
    dim_1 = data_arr.shape[1]

    x = np.zeros(( dim_0, time_steps, dim_1))
    y = np.zeros((dim_0))

    for i in range(dim_0 - 1):
        x[i] = data_arr[i: i + time_steps]
        y[i] = data_arr[i + 1, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y

#%%
# the DataSet has to be dividible through the Batch size

def trim_data_set(data_arr, batch_size):

    rows_drop = data_arr.shape[0] % batch_size

    if rows_drop > 0:
        return data_arr[:- rows_drop ]
    else:
        return data_arr
    
#%%
# Build the training and test data in the correct manner for LSTM
# The Input Array for the keras LSTM Model needs 3 Dimensions.
# (data, time_step, features)
 
time_step = 20
batch_size = 100

x_train, y_train = build_time_series(train, 3, time_step)
x_train = trim_data_set(x_train, batch_size)
y_train = trim_data_set(y_train, batch_size)

x_test, y_test = build_time_series(test, 3, time_step)
x_test = trim_data_set(x_test, batch_size)
y_test = trim_data_set(y_test, batch_size)

#%%
# Build up the LSTM Model

lstm_model = Sequential()
lstm_model.add(LSTM(1, batch_input_shape=(batch_size, \
    time_step, x_train.shape[2] ), dropout=0.0, recurrent_dropout=0.0, \
        stateful=True, kernel_initializer='random_uniform' ))
#lstm_model.add(Dropout(0.5))
#lstm_model.add(Dense(20,activation='sigmoid'))
#lstm_model.add(Dense(1, activation='linear'))

# Select Optimizer
model_optimizer = Adam(lr=0.001)
lstm_model.compile(loss='mean_squared_error', optimizer=model_optimizer)
#%%
# fit the model

# logger is missing here
csv_logger = CSVLogger(output_path + r"\log.csv", append=True, separator=";")
checkpoint_logger = ModelCheckpoint(output_path + r"\weights.h5", monitor='acc', verbose=0,\
     save_best_only=True, save_weights_only=False, mode='auto', period=1 )

history = lstm_model.fit(x_train,y_train,epochs=5, verbose=1,\
    batch_size=batch_size, shuffle=False, callbacks=[checkpoint_logger])

#%%


results = lstm_model.predict(x_test, batch_size=batch_size)
print(r2_score(y_test, results))
plt.plot(results)
plt.plot(y_test)
plt.show()



#%%
