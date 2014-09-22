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
    
    
    day_in_sec = 1440
    num_of_samples = 300
    offset = 600e6
    num_of_pred = 100
    observations = data[:num_of_samples,1].tolist()
    x = data[:num_of_samples+num_of_pred,0]-offset
    x = x / 1e6
    x = x.tolist()
        
    forecast, alpha, beta, gamma, rmse = hw.additive(observations, m=288, fc=num_of_pred)
    
    prediction = []
    for _ in range(0,len(observations)):
        prediction.append(None)
        
    for i in range(0, len(forecast)):
        prediction.append(forecast[i])
        observations.append(None)
    
    data = data[:num_of_samples+num_of_pred,1].tolist()
        
    plt.plot(x, data)
    plt.plot(x, prediction)
    plt.show()