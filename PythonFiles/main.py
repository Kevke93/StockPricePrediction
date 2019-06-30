# %%
# main Function for the Trading Enviroment
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.append(r"C:\Users\admin\Desktop\Masterarbeit\Programm\StockPricePrediction\PythonFiles")

from Data_Processing import inital_clearing, build_data_LSTM
from Model_LSTM import LSTM_MODEL_PURE
from Trading_Engine import trading_engine
# Select the Data
DATA_FOLDER = Path(os.getcwd()).parents[0]
DATA_FILE = "StockPricePrediction\Data\DAX30.csv"
DATA_PATH = os.path.join(DATA_FOLDER,DATA_FILE)
FEATURES = ['Open', 'High', 'Low', 'Close']
Y_COL_INDEX = 3
BATCH_SIZE = 100
TIME_STEPS = 20
TRAIN_SIZE = 0.8
PRICE_INDEX_PERCENTAGE = 0.00001

#Returns a DATAFRAME all the Data

df_data = inital_clearing(DATA_PATH, FEATURES)

#Scal the Data Between 0 and 1

scaler = MinMaxScaler(feature_range=(0,1))
scalerY = MinMaxScaler(feature_range=(0,1))
scalerY.fit(df_data.Close.values.reshape(-1, 1))

# Split the Data
df_train, df_test = train_test_split(df_data, shuffle=False, train_size=TRAIN_SIZE,\
    test_size=abs(1-TRAIN_SIZE) )

df_train = scaler.fit_transform(df_train)
df_test = scaler.transform(df_test)
# Create the Data for the Model
x_train, y_train, x_test, y_test = build_data_LSTM(df_train, df_test,\
    Y_COL_INDEX, TIME_STEPS, BATCH_SIZE)

#scale the Data

# Select the Model
model = LSTM_MODEL_PURE()
model.build_fit(BATCH_SIZE, TIME_STEPS, x_train, y_train,)

# Initialize the Trading Enginge

trader = trading_engine(model, PRICE_INDEX_PERCENTAGE, scalerY)
trader.trade(x_test, y_test)

