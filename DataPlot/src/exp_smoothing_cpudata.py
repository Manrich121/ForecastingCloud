from multiprocessing import Pool as ThreadPool 

from sklearn.metrics import mean_squared_error, r2_score

import HW_model
import fileutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def performsSlidingWindowForecast(filename, minpercentile=5, step=30, input_window=3000, predic_window=30):
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
    fileutils.writeCSV("D:/Wikipage data/pageviews_hw/"+f, np.atleast_2d(result))
    print filename, "complete!"

if __name__ == '__main__':
    aggregatedRmse = None
    pool = ThreadPool(4)
    files =  fileutils.getFilelist("D:/Wikipage data/pageviews")
    
#     performsSlidingWindowForecast(files[1])
    pool.map(performsSlidingWindowForecast, files)
    pool.close()
    pool.join()