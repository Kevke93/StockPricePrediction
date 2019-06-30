import pandas as pd 
import numpy as np
import os
from keras.optimizers import Adam
from keras.models import Sequential
from Data_Processing import build_time_series_LSTM, trim_data_set
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import LSTM
import time
from pathlib import Path

class LSTM_MODEL_PURE:
    def __init__(self):
        relativ_path = r"Logs\LSTM_PURE"
        self.log_path = os.path.join(os.getcwd(), relativ_path)

    def build_fit(self, BATCH_SIZE, TIME_STEPS, x_train, y_train, epochs=5):

        self.BATCH_SIZE = BATCH_SIZE
        # Buil the LSTM Model, For the PURE LSTM Model JUST 1 Output

        self.model = Sequential()
        self.model.add(LSTM(1, batch_input_shape=(BATCH_SIZE, \
             TIME_STEPS, x_train.shape[2] ), dropout=0.0, recurrent_dropout=0.0, \
            stateful=True, kernel_initializer='random_uniform' ))
        model_optimizer = Adam(lr=0.001)
        self.model.compile(loss='mean_squared_error', optimizer=model_optimizer)

        self.csv_logger = CSVLogger(self.log_path + r"\log.time.ctime().replace(' ','_').csv",\
                         append=True, separator=";")
        checkpoint_logger = ModelCheckpoint(self.log_path + r"\weights.{epoch:02d}-\
                        .time.ctime().replace(' ','_').h5",\
                        verbose=0, save_best_only=True,\
                        save_weights_only=False, mode='auto', period=1, monitor='val_accuracy')
        
        self.model.fit(x_train, y_train, epochs=epochs, verbose=1,\
                    batch_size=BATCH_SIZE, shuffle=False, callbacks=[checkpoint_logger])

    def predict(self, x_test):
        return self.model.predict(x_test, batch_size=self.BATCH_SIZE)
        
    #def fit_optimaze_parameters():
    #def load_model():
