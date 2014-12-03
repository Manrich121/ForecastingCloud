import fileutils
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa as sm
import statsmodels
from sklearn.metrics import mean_squared_error, r2_score
from Wavelet_model import Wavelet_model

from multiprocessing import Pool as ThreadPool 

def performsSlidingWindowForecast(filename, minpercentile=5, step=30, input_window=3000, predic_window=30):
    '''
    Input window = 250 hours = 250*12 = 3000 
    look ahead window 60 samples =  5 hours = 720min/5 = 60
    '''
    data = np.nan_to_num(np.genfromtxt(filename, delimiter=',', skip_header=1))
    minimum = np.percentile(data[:,1],minpercentile)
    N = len(data[:,1])
    result = []
    print filename, "started..."
    for strIndex in range(0,N-input_window - predic_window, step):
        if strIndex == 0:
            y = data[:input_window,1]
            model = Wavelet_model(y)
            model.fit()
        else:
            y = data[strIndex:strIndex+input_window,1]
            model.update(y)
              
        y_pred = model.predict(predic_window)
        y_pred[y_pred<0] = minimum
        result.append(y_pred)
    f = filename.split('/')[-1]
    fileutils.writeCSV("d:/data/cpu_wavelet_forecasts/"+f, np.atleast_2d(result))
    print filename, "complete!"

if __name__ == '__main__':
    aggregatedRmse = None
    pool = ThreadPool(4)
    files =  fileutils.getFilelist("D:/data/cpuRate")
    
    performsSlidingWindowForecast("D:/data/cpuRate/cpu_1095481.csv")
#     pool.map(performsSlidingWindowForecast, files[:1])
    pool.close()
    pool.join()