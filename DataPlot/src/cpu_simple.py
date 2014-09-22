'''
Created on 22 Sep 2014

@author: Manrich
'''
import fileutils
import numpy as np
from matplotlib import pyplot as plt
import holtwinters as hw


if __name__ == '__main__':
    data = np.genfromtxt("d:/data/cpuRate/cpuRate_4815459946.csv", delimiter=',', usecols=(0,1))
    '''
        Input window = 250 hours = 250*12 = 3000 
        look ahead window = 5 hours = 5*12 = 60
        one day sample = 24hours*60min /5min = 288
    '''
    input_window = 3000
    predic_window = 60
    one_day_samples = 288
    
    Y = data[:input_window,1].tolist()
    predictions, alpha, beta, gamma, rmse = hw.additive(Y, m=one_day_samples, fc=predic_window)
    
    real = data[input_window:input_window+predic_window,1]
    
    plt.plot(real)
    plt.plot(predictions)
   
    plt.show()