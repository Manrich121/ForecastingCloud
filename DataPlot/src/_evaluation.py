import fileutils
import evaluation as eval
import numpy as np
import os
from multiprocessing import Pool as ThreadPool 

METHOD = "press"
TYPE = "cpu2"

def performEvaluations(filename, train_window = 3000, overload_dur = 5, overload_percentile = 70, steps=30):
        
    cur_results = []
    forecasts = np.nan_to_num(np.genfromtxt("d:/data/"+TYPE+"_"+METHOD+"/" + filename, delimiter=',',usecols=range(0,30))).ravel() # ,usecols=range(0,30)
    truevals = np.genfromtxt("d:/data/"+TYPE+"/"+filename, delimiter=',',skip_header=1)[:train_window+len(forecasts),1]
    
    # Normalize
#     truevals = np.divide(truevals, np.max(truevals))
    
    threshold =  np.percentile(truevals, overload_percentile)
    
    cur_results.append(eval.calc_RMSE(truevals[train_window:], forecasts))
    for val in eval.calc_upper_lower_acc(truevals[train_window:], forecasts):
        cur_results.append(val) 
    for val in eval.calc_persample_accuracy(truevals[train_window:], forecasts, threshold):
        cur_results.append(val)
    for val in eval.calc_overload_states_acc(truevals[train_window:], forecasts, threshold):
        cur_results.append(val)
        
    return cur_results

if __name__ == '__main__':
    files = []
    
    root = "d:/data/"+TYPE+"/"
    for _, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".csv"):
                files.append(f)          
    pool = ThreadPool(4)
    
#     performEvaluations(files[0])
    results = pool.map(performEvaluations, files)
    pool.close()
    pool.join()
     
    fileutils.writeCSV("d:/data/results/"+TYPE+"_"+METHOD+".csv", results)
    print METHOD+" "+ TYPE + " complete"