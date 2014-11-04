
# coding: utf-8

# In[1]:
import fileutils
import numpy as np
import matplotlib.pyplot as plt
import holtwinters as hw
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

def insertIntoArray(aggregatedRmse, row):
    if aggregatedRmse == None:
        return np.atleast_2d(row)
    else:
        return np.vstack((aggregatedRmse, np.array(row)))
        

# In[2]:

def performsSlidingWindowForecast(filename, input_window=3000, predic_window=60):
    '''
    Input window = 250 hours = 250*12 = 3000 
    look ahead window 60 samples =  5 hours = 720min/5 = 60
    '''
    data = pd.read_csv(filename)
    N = len(data.Cpu)
    forecastRmse = []
    
    for strIndex in range(0,N-input_window - predic_window,predic_window):
        print strIndex       
        y = data.Cpu[strIndex:strIndex + input_window].tolist()
        y_true = data.Cpu[strIndex + input_window:strIndex + input_window+predic_window]

        params, rmse = hw.fit(y)
        
        y_pred = hw.predict(y, fc=predic_window, modelparams=params)
        forecastRmse.append(mean_squared_error(y_true, y_pred))
        
    print filename, "complete!"
    return forecastRmse

if __name__ == '__main__':
    aggregatedRmse = None
    count =0
    pool = ThreadPool(3)
    files =  fileutils.getFilelist("D:/data/cpuRate")
    
    results = pool.map(performsSlidingWindowForecast, files[:3])
    pool.close()
    pool.join()
    # for f in files:
    #     count+=1
    #     print count, f
    #     aggregatedRmse = insertIntoArray(aggregatedRmse, performsSlidingWindowForecast(f))
    
    print results
#     fileutils.writeCSV("d:/data/cpuRate/output.csv",aggregatedRmse)
    
    # print "Model rmse:", rmse
    # print "Forecast RMSE:", mean_squared_error(y_true, y_pred)
    # print "R sqaured:", r2_score(y_true, y_pred)

