# %matplotlib inline
import fileutils
import evaluation as eval
import numpy as np
import os
from multiprocessing import Pool as ThreadPool 

METHOD = "hw"
TYPE = "pageviews"

def performEvaluations(filename, train_window = 3000, overload_dur = 5, overload_percentile = 70, steps=30):
        
    cur_results = []
    filename
    forecasts = np.nan_to_num(np.genfromtxt("d:/Wikipage data/"+TYPE+"_"+METHOD+"/" + filename, delimiter=',',usecols=range(0,30))).ravel() # 
#     truevals = np.genfromtxt("d:/Wikipage data/"+TYPE+"/"+filename, delimiter=',',skip_header=1)[:train_window+len(forecasts),1]
 
    truevals = np.genfromtxt("d:/Wikipage data/"+TYPE+"/"+filename)[:train_window+len(forecasts)]
    truevals = truevals/np.max(truevals)
        
    cur_results.append(eval.calc_RMSE(truevals[train_window:], forecasts))
    for val in eval.calc_upper_lower_acc(truevals[train_window:], forecasts):
        cur_results.append(val) 
            
    return cur_results

if __name__ == '__main__':
    files = []
    pool = ThreadPool(4)
    
    root = "d:/Wikipage data/"+TYPE+"/"
    for _, _, fs in os.walk(root):
        for f in fs:
            files.append(f)  
    
#     performEvaluations(files[0])
    results = pool.map(performEvaluations, files)
    pool.close()
    pool.join()
     
    fileutils.writeCSV("d:/Wikipage data/results/"+TYPE+"_"+METHOD+".csv", results)
    print METHOD+" "+ TYPE + " complete"