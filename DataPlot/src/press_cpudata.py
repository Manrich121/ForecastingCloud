import fileutils
import numpy as np
import matplotlib.pyplot as plt
import Press_model

from multiprocessing import Pool as ThreadPool 

def performsSlidingWindowForecast(filename, minpercentile=5, step=30, input_window=3000, predic_window=30):
    '''
    Input window = 250 hours = 250*12 = 3000 
    look ahead window 60 samples =  5 hours = 720min/5 = 60
    '''
#     Wikidata
#     data = np.genfromtxt(filename)
#     data = data/np.max(data)

    data = np.nan_to_num(np.genfromtxt(filename, delimiter=',', skip_header=1)[:,1]).ravel()
    minimum = np.percentile(data,minpercentile)
    N = len(data)
    result = []
    max = np.max(data)
    print filename, "started..."
    for strIndex in range(0,N-input_window - predic_window, step):
        if strIndex == 0:
            y = data[:input_window]
            model = Press_model.Press_model(y, maximum=max)
            model.fit()
        else:
#             y = data[strIndex:strIndex+input_window,1]
            y = data[input_window + strIndex - step:input_window + strIndex]
            model.update(y)
        y_pred = model.predict(predic_window)
        y_pred[y_pred<0] = minimum
        result.append(y_pred)
    res = np.zeros((len(result),predic_window))
    
    for i in range(len(result)):
        res[i,:len(result[i])] = result[i] 
    f = filename.split('/')[-1]
    fileutils.writeCSV("d:/data/cpu2_press/"+f, np.atleast_2d(res))
    print filename, "complete!"

if __name__ == '__main__':
    aggregatedRmse = None
    pool = ThreadPool(4)
    files =  fileutils.getFilelist("D:/data/cpu2")
    pool.map(performsSlidingWindowForecast, files)
    pool.close()
    pool.join()