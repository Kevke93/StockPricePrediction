import os
import sys
import pandas as pd
from Model_LSTM import LSTM_MODEL_PURE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
class trading_engine:
    def __init__(self, model, PRICE_INDEX_PERCENTAGE, scalerY):
        self.PRICE_INDEX_PERCENTAGE = PRICE_INDEX_PERCENTAGE
        self.model = model
        self.scalerY = scalerY


    def trade(self, x_test, y_test):
        predictions = self.model.predict(x_test)
        print(predictions)
        predictions_unscaled = self.scalerY.inverse_transform(predictions)
        y_test_unscaled = self.scalerY.inverse_transform(y_test.reshape(-1, 1))

        plt.plot(predictions)
        plt.plot(y_test)
        plt.show()

        #for i in range(len(predictions)):
         #   if abs(predictions[i] - ) > self.PRICE_INDEX_PERCENTAGE:

    
    

