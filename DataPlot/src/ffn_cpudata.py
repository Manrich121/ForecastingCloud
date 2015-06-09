from multiprocessing import Pool as ThreadPool 

from sklearn.metrics import mean_squared_error, r2_score
import statsmodels

import Fnn_model
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
    curMachine = filename.split('/')[-1]
    model = Fnn_model.Fnn_model(data=data, machineID = curMachine,netPath='../data/cpu2_networks/'+curMachine.replace(".csv",".xml"), eta=curEta, lmda=curLmda)
    model.fit()
    
    pred = []
#     lastFc = 0
    for p in range(input_window, len(data)-predic_window,predic_window):
        fc = np.array(model.predict(predic_window)).flatten()
#         fc[-1] = lastFc
#         lastFc = fc[0]
        fc[fc<0] = minimum
        pred.append(fc)
        model.update()
    
    pred = np.array(pred)
        
    f = filename.split('/')[-1]
    fileutils.writeCSV("d:/data/cpu2_fnn/"+f, np.atleast_2d(pred))
    print filename, "complete!"

if __name__ == '__main__':
    root = "D:/data/cpu2/"
    pool = ThreadPool(4)
    hyperparms =  np.genfromtxt("..\data\cpu2_networks\hyperparams.csv", delimiter=',', dtype=None)
    files_etas_lmads = []
    count =0
    for curRow in hyperparms:
        files_etas_lmads.append([root+curRow[0].strip("'")+".csv", curRow[3], curRow[4]])
        
    performsSlidingWindowForecast(files_etas_lmads[0])
#     pool.map(performsSlidingWindowForecast, files_etas_lmads)
#     pool.close()
#     pool.join()