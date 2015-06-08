from multiprocessing import Pool as ThreadPool 

from sklearn.metrics import mean_squared_error, r2_score
import statsmodels

import Rnn_model
import fileutils
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa as sm


def performsSlidingWindowForecast(filename_eta_lmda, minpercentile=5, step=30, input_window=3000, predic_window=30):
    '''
    Input window = 250 hours = 250*12 = 3000 
    look ahead window 60 samples =  5 hours = 720min/5 = 60
    '''
    filename, curEta, curLmda = filename_eta_lmda
    data = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(1))
    minimum = np.percentile(data,minpercentile)
    N = len(data)
    
    print filename, "started..."
    curMachine = filename.split('/')[-1][:-4]
#     curMachine = "cpu" + filename.split('/')[-1].strip('.csv')
    
    model = Rnn_model.Rnn_model(data=data, machineID = curMachine, eta=curEta, lmda=curLmda)
    model.fit()
    
    pred = []
    lastFc = None
    for p in range(input_window, len(data)-predic_window,predic_window):
        fc = model.predict(predic_window)
        if lastFc is not None:
            fc[0] = lastFc 
        lastFc = fc[-1]
        fc[fc<0] = minimum
        pred.append(fc)
        model.update()
    
    pred = np.array(pred).ravel()    
    f = filename.split('/')[-1]
    fileutils.writeCSV("d:/data/cpu_rnn_forecasts/"+f, np.atleast_2d(pred))
    print filename, "complete!"

if __name__ == '__main__':
    root = "D:/data/cpuRate/"
    pool = ThreadPool(4)
    hyperparms =  np.genfromtxt("..\data\cpu_rnn_networks\hyperparams.csv", delimiter=',', dtype=None)
    
    files_etas_lmads = []
    count = 0
    for curRow in hyperparms:
        files_etas_lmads.append([root+curRow[0].strip("'")+".csv", curRow[3], curRow[4]])
    performsSlidingWindowForecast(files_etas_lmads[0])
#     pool.map(performsSlidingWindowForecast, files_etas_lmads)
#     pool.close()
#     pool.join()