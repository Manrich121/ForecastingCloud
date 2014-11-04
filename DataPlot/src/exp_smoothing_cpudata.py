import fileutils
import numpy as np
import matplotlib.pyplot as plt
import HW_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

from multiprocessing.dummy import Pool as ThreadPool 

def performsSlidingWindowForecast(filename, input_window=3000, predic_window=60):
    '''
    Input window = 250 hours = 250*12 = 3000 
    look ahead window 60 samples =  5 hours = 720min/5 = 60
    '''
    data = pd.read_csv(filename)
    N = len(data.Cpu)
    result = []
    print filename, "started..."
    for strIndex in range(0,N-input_window - predic_window,predic_window):   
         
        y = data.Cpu[strIndex:strIndex + input_window].tolist()
        y_true = data.Cpu[strIndex + input_window:strIndex + input_window+predic_window]

        model = HW_model.HW_model(y, 'additive')
        params, rmse = model.fit()
                
        y_pred = model.predict(fc=predic_window)
        result.append(y_pred)
    print filename, "complete!"
    f = filename.split('/')[-1]
    fileutils.writeCSV("d:/data/cpu_exp_smooting_forecasts/"+f, np.atleast_2d(result))

if __name__ == '__main__':
    aggregatedRmse = None
    pool = ThreadPool(4)
    files =  fileutils.getFilelist("D:/data/cpuRate")
    
    pool.map(performsSlidingWindowForecast, files)
    pool.close()
    pool.join()
    # for f in files:
    #     count+=1
    #     print count, f
    #     aggregatedRmse = insertIntoArray(aggregatedRmse, performsSlidingWindowForecast(f))

