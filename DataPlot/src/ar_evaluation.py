# %matplotlib inline
import fileutils
import tsutils
import evaluation as eval
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from multiprocessing import Pool as ThreadPool 

def performEvaluations(filename, train_window = 3000, overload_dur = 5, overload_percentile = 70, steps=30):
    cur_results = []
    forecasts = np.genfromtxt("d:/data/memory_ar_forecasts/"+ filename,delimiter=',',usecols=range(0,steps)).ravel()
    truevals = np.genfromtxt("d:/data/memory/"+filename, delimiter=',',skip_header=1)[train_window:train_window+len(forecasts),1]
    
    threshold =  np.percentile(truevals, overload_percentile)
    
    cur_results.append(eval.calc_RMSE(truevals, forecasts))
    for val in eval.calc_upper_lower_acc(truevals, forecasts):
        cur_results.append(val) 
    for val in eval.calc_persample_accuracy(truevals, forecasts, threshold):
        cur_results.append(val)
    for val in eval.calc_overload_states_acc(truevals, forecasts, threshold):
        cur_results.append(val)
        
    return cur_results

if __name__ == '__main__':
    files = []
    for _, _, fs in os.walk("d:/data/memory/"):
        for f in fs:
            if f.endswith(".csv"):
                files.append(f)
    
    pool = ThreadPool(4)
    results = pool.map(performEvaluations, files)
    pool.close()
    pool.join()
    
    fileutils.writeCSV("d:/data/results/memory_autoregressive.csv", results)