import fileutils
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa as sm
import statsmodels
from sklearn.metrics import mean_squared_error, r2_score
import AR_model

from multiprocessing import Pool as ThreadPool 

TYPE = "cpu"

def performsSlidingWindowForecast(filename, minpercentile=5, step=30, input_window=3000, predic_window=30):
    '''
    Input window = 250 hours = 250*12 = 3000 
    look ahead window 60 samples =  5 hours = 720min/5 = 60
    '''
    data = np.genfromtxt(filename, delimiter=',',skip_header=1)
    # Normalize data
#     data[:,1] = np.divide(data[:,1], np.max(data[:,1]))
    
    minimum = np.percentile(data[:,1],minpercentile)
    N = len(data[:,1])
    result = []
    print filename, "started..."
    for strIndex in range(0,N-input_window - predic_window, step):
        if strIndex == 0:
            y = data[:input_window,1]
            model = AR_model.AR_model(y, order=30)
            model.fit()
        else:
            y = data[strIndex:strIndex+input_window,1]
            model.update(y)
              
        y_pred = model.predict(predic_window)
        y_pred[y_pred[:,0]<0,0] = minimum
        result.append(y_pred[:,0])
    f = filename.split('/')[-1]
    fileutils.writeCSV("d:/data/"+TYPE+"_ar_forecasts/"+f, np.atleast_2d(result))
    print filename, "complete!"

if __name__ == '__main__':
    aggregatedRmse = None
    pool = ThreadPool(4)
    files =  fileutils.getFilelist("D:/data/"+TYPE)
#     print files
#     performsSlidingWindowForecast(files[0])
    pool.map(performsSlidingWindowForecast, files)
    pool.close()
    pool.join()