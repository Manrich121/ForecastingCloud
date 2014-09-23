'''
Created on 22 Sep 2014

@author: Manrich
'''
import fileutils
import numpy as np
from matplotlib import pyplot as plt
import holtwinters as hw
from scipy import fft, arange


if __name__ == '__main__':
    data = np.genfromtxt("d:/data/cpuRate/cpuRate_4815459946.csv", delimiter=',', usecols=(0,1))
    '''
        Input window = 250 hours = 250*12 = 3000 
        look ahead window 60 samples =  5 hours = 5*12 = 60
        one day sample = 24hours*60min /5min = 288
    '''
    input_window = 3000
    predic_window = 60
    one_day_samples = 288
    real = data[input_window:input_window+predic_window,1]
    

#     fs = 1/300.0
#     n = input_window
#     k = arange(n)
#     T = n/fs
#     frq = k/T
#     frq = frq[range(n/2)]
#     
#     fourier = fft(data[:input_window,1])/n
#     fourier = fourier[range(n/2)]
# #     plt.plot(frq,abs(fourier),'r')
#     print abs(fourier)
        
    Y = data[:input_window,1].tolist()
    forecast, alpha, beta, gamma, rmse = hw.additive(Y, m=300, fc=predic_window)
    
    plt.plot(real)
    plt.plot(forecast)
    plt.show()
    
    