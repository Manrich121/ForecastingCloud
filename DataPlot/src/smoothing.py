'''
Created on 19 Sep 2014

@author: Manrich
'''

from matplotlib import pyplot as plt

import holtwinters as hw
import numpy as np


data = [362, 385,432, 341, 382, 409,
        498, 387, 473, 513, 582, 474,
        544, 582, 681, 557, 628, 707,
        773, 592, 627, 725, 854, 661]


if __name__ == '__main__':

    forecastCount = 12
    x = np.array([i for i in range(0, len(data)+forecastCount)])
    linearfc = np.zeros([1,len(data)+forecastCount])
    additivefc = np.zeros([1,len(data)+forecastCount])
    multiplefc = np.zeros([1,len(data)+forecastCount])
    for i in range(2, len(data)):
        forecast, alpha, beta, rmse = hw.linear(data[:i], 1)
        linearfc[0, i:i+len(forecast)] = forecast
        
        if i>4:
            forecast, alpha, beta, gamma, rmse = hw.additive(data[:i], m=4, fc=1)
            additivefc[0, i:i+len(forecast)] = forecast
            
            forecast, alpha, beta, gamma, rmse = hw.multiplicative(data[:i], m=4, fc=1)
            multiplefc[0, i:i+len(forecast)] = forecast
        else:
            forecast, alpha, beta, gamma, rmse = hw.additive(data[:i], m=1, fc=1)
            additivefc[0, i:i+len(forecast)] = forecast
            
            forecast, alpha, beta, gamma, rmse = hw.multiplicative(data[:i], m=1, fc=1)
            multiplefc[0, i:i+len(forecast)] = forecast
            
        
#     Linear
    forecast, alpha, beta, rmse = hw.linear(data, forecastCount)
    linearfc[0,-forecastCount:] = forecast

#     Additive
    forecast, alpha, beta, gamma, rmse = hw.additive(data, m=4, fc=forecastCount)
    additivefc[0, -forecastCount:] = forecast
#     Multiply
    forecast, alpha, beta, gamma, rmse = hw.multiplicative(data, m=4, fc=forecastCount)
    multiplefc[0, -forecastCount:] = forecast
   
    for _ in range(0,forecastCount):
        data.append(None)
    data = np.array(data)

    
    plt.figure()
    plt.plot(x, data, c="k")
    plt.plot(x, linearfc[0], c="r")
    plt.plot(x, additivefc[0], c="b")
#     plt.plot(x, multiplefc[0], c="g")
    plt.show()