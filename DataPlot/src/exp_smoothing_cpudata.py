import fileutils
import numpy as np
import matplotlib.pyplot as plt
import HW_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

from multiprocessing import Pool as ThreadPool 

def performsSlidingWindowForecast(filename, minpercentile=5, step=30, input_window=3000, predic_window=60):
    '''
    Input window = 250 hours = 250*12 = 3000 
    look ahead window 60 samples =  5 hours = 720min/5 = 60
    '''
    data = np.genfromtxt(filename)
    data = data/np.max(data)
    minimum = np.percentile(data,minpercentile)
    N = len(data)
    result = []
    print filename, "started..."
    y = data[0:input_window].tolist()
    model = HW_model.HW_model(y, minimum, 'additive')
    
    for strIndex in range(0,N-input_window - predic_window, step):
        if strIndex == 0:
            model.fit()
        else:
            y = data[strIndex:strIndex+input_window].tolist()
            model.update(y)
              
        y_pred = model.predict(fc=predic_window)
        result.append(y_pred)
    
    f = filename.split('/')[-1]
    fileutils.writeCSV("D:/Wikipage data/network_hw/"+f, np.atleast_2d(result))
    print filename, "complete!"

if __name__ == '__main__':
    aggregatedRmse = None
    pool = ThreadPool(4)
    files =  fileutils.getFilelist("D:/Wikipage data/network")
    
#     performsSlidingWindowForecast(files[1])
    pool.map(performsSlidingWindowForecast, files)
    pool.close()
    pool.join()