
# coding: utf-8

# In[1]:
import fileutils
import numpy as np
import matplotlib.pyplot as plt
import holtwinters as hw
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


# In[2]:

def performsSlidingWindowForecast(filename, input_window, predic_window):
    '''
    Input window = 250 hours = 250*12 = 3000 
    look ahead window 60 samples =  5 hours = 720min/5 = 60
    '''
    data = pd.read_csv(filename)
    N = len(data.Cpu)
    forecastRmse = []
    
    print filename, "started"
    
    for strIndex in range(0,N-input_window - predic_window,predic_window):
               
        y = data.Cpu[strIndex:strIndex + input_window].tolist()
        y_true = data.Cpu[strIndex + input_window:strIndex + input_window+predic_window]

        params, rmse = hw.fit(y)
        
        y_pred = hw.predict(y, fc=predic_window, modelparams=params)
        forecastRmse.append(mean_squared_error(y_true, y_pred))
        
    print filename, "complete!"
    fileutils.writeCSV("D:/data/test.csv", data=[forecastRmse], mode='a')


# In[ ]:

files =  fileutils.getFilelist("D:/data/cpuRate")

for f in files:
    performsSlidingWindowForecast(f, 3000, 60)

# print "Model rmse:", rmse
# print "Forecast RMSE:", mean_squared_error(y_true, y_pred)
# print "R sqaured:", r2_score(y_true, y_pred)

