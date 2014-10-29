# Holt-Winters algorithms to forecasting
# Coded in Python 2 by: Andre Queiroz
# Description: This module contains three exponential smoothing algorithms. They are Holt's linear trend method and Holt-Winters seasonal methods (additive and multiplicative).
# References:
#  Hyndman, R. J.; Athanasopoulos, G. (2013) Forecasting: principles and practice. http://otexts.com/fpp/. Accessed on 07/03/2013.
#  Byrd, R. H.; Lu, P.; Nocedal, J. A Limited Memory Algorithm for Bound Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5, pp. 1190-1208.
 
from sys import exit
from math import sqrt
from numpy import array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import statsmodels.api as sm

def fit(y, type='additive'):
    '''
    Fit a Holt-winters method of type={linear, additive, multiplicative} on data
    
    Parameters
    ----------
    y: Data/Observation
    type: String specifying the type of method to use
    
    Return
    ----------
    params: Array-like Parameters of the model [alpha, beta, gamma, m] depending on the type
    RMSE: Root Mean Squared Error of the model when fit to the data
    '''
    alpha, beta, gamma = 0.0, 0.0, 0.0
    rmse = 0.0
    
    if type == 'linear':
        initial_values = array([0.3, 0.1])
        boundaries = [(0, 1), (0, 1)]
 
        parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (y, type), bounds = boundaries, approx_grad = True)
        alpha, beta = parameters[0]
        rmse = parameters[1]
        return [alpha, beta], rmse
        
    elif type == 'additive':
        m = findDominentSeason(y)
     
        initial_values = array([0.3, 0.1, 0.1])
        boundaries = [(0, 1), (0, 1), (0, 1)]
 
        parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (y, type, m), bounds = boundaries, approx_grad = True)
        alpha, beta, gamma = parameters[0]
        rmse = parameters[1]
        
        return [alpha, beta, gamma, m], rmse
        
    elif type == 'multiplicative':
        m = findDominentSeason(y)
        
        initial_values = array([0.0, 1.0, 0.0])
        boundaries = [(0, 1), (0, 1), (0, 1)]
 
        parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (y, type, m), bounds = boundaries, approx_grad = True)
        alpha, beta, gamma = parameters[0]
        rmse = parameters[1]
        
        return [alpha, beta, gamma, m], rmse

def predict(y, fc, modelparams, type='additive'):
    '''
    Forecasts fc samples into the future, using the model parameters corresponding to the model type
    
    Parameters
    ----------
    
    y: Observations to what the model was fitted
    fc: Number of samples ahead to predict
    modelparams: Array-like [alpha, gamma, beta, m] model parameters dependent on the model type
    type: String specifying the type of method to use type={linear, additive, multiplicative}
    
    Return
    ----------
    Forecast: List of predicted values of length fc
    '''
    Y = y[:]
    if type == 'linear':
        alpha, beta = modelparams[0], modelparams[1]
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        y = [a[0] + b[0]]
     
        for i in range(len(Y) + fc):
            if i == len(Y):
                Y.append(a[-1] + b[-1])
     
            a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] + b[i + 1])
            
    elif type == 'additive':
        alpha, beta, gamma, m = modelparams
        a = [sum(Y[0:m]) / float(m)]
        b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
        s = [Y[i] - a[0] for i in range(m)]
        y = [a[0] + b[0] + s[0]]
     
        for i in range(len(Y) + fc):
            if i == len(Y):
                Y.append(a[-1] + b[-1] + s[-m])
     
            a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
            y.append(a[i + 1] + b[i + 1] + s[i + 1])
            
    elif type == 'multiplicative':
        alpha, beta, gamma, m = modelparams
        a = [sum(Y[0:m]) / float(m)]
        b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
        s = [Y[i] / a[0] for i in range(m)]
        y = [(a[0] + b[0]) * s[0]]
     
        for i in range(len(Y) + fc):
     
            if i == len(Y):
                Y.append((a[-1] + b[-1]) * s[-m])
     
            a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
            y.append((a[i + 1] + b[i + 1]) * s[i + 1])
            
    return Y[-fc:]

def linear(x, fc, alpha = None, beta = None):
    '''
    Linear Holt method for data with a trend
    
    Parameters
    ----------
    x: Observations
    fc: Number of samples ahead to predict
    alpha: Smoothing constant, will be estimate if None
    beta: Trend smoothing constant, will be estimated if None

    Return
    ----------
    Forecast, alpha, beta, RMSE
    '''
 
    Y = x[:]
 
    if (alpha == None or beta == None):
 
        initial_values = array([0.3, 0.1])
        boundaries = [(0, 1), (0, 1)]
        type = 'linear'
 
        parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type), bounds = boundaries, approx_grad = True)
        alpha, beta = parameters[0]
 
    a = [Y[0]]
    b = [Y[1] - Y[0]]
    y = [a[0] + b[0]]
    rmse = 0
 
    for i in range(len(Y) + fc):
 
        if i == len(Y):
            Y.append(a[-1] + b[-1])
 
        a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        y.append(a[i + 1] + b[i + 1])
 
    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
 
    return Y[-fc:], alpha, beta, rmse
  

def additive(x, fc, alpha = None, beta = None, gamma = None):
    '''
    Holt-Winters method for data with trend and (roughly constant) seasonal variation
    
    Parameters
    ----------
    x: Observations
    m: Number of seasons in data
    fc: Number of samples ahead to predict
    alpha: Smoothing constant, will be estimate if None
    beta: Trend smoothing constant, will be estimated if None
    gamma: Seasonal smoothing constant, will be estimated if None
    
    Return
    ---------
    Forecast, alpha, beta, gamma, RMSE
    '''
    
#     Determine dominant season length in samples
    m = findDominentSeason(x)
    Y = x[:]
 
    if (alpha == None or beta == None or gamma == None):
 
        initial_values = array([0.3, 0.1, 0.1])
        boundaries = [(0, 1), (0, 1), (0, 1)]
        type = 'additive'
 
        parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, m), bounds = boundaries, approx_grad = True)
        alpha, beta, gamma = parameters[0]
 
#  Forecasting part:
    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
    s = [Y[i] - a[0] for i in range(m)]
    y = [a[0] + b[0] + s[0]]
    rmse = 0
 
    for i in range(len(Y) + fc):
 
        if i == len(Y):
            Y.append(a[-1] + b[-1] + s[-m])
 
        a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
        y.append(a[i + 1] + b[i + 1] + s[i + 1])
 
    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
 
    return Y[-fc:], alpha, beta, gamma, rmse

def multiplicative(x, m, fc, alpha = None, beta = None, gamma = None):
    '''
    Holt-Winters method for data with trend and seasonal variation that change proportional to the level of the series
    x: Observations
    m: Number of seasons in data
    fc: Number of samples ahead to predict
    alpha: Smoothing constant, will be estimate if None
    beta: Trend smoothing constant, will be estimated if None
    gamma: Seasonal smoothin constant, will be estimated if None
    
    return return: Forecast, alpha, beta, gamma, RMSE
    '''
 
    Y = x[:]
 
    if (alpha == None or beta == None or gamma == None):
 
        initial_values = array([0.0, 1.0, 0.0])
        boundaries = [(0, 1), (0, 1), (0, 1)]
        type = 'multiplicative'
 
        parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, m), bounds = boundaries, approx_grad = True)
        alpha, beta, gamma = parameters[0]
 
    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
    s = [Y[i] / a[0] for i in range(m)]
    y = [(a[0] + b[0]) * s[0]]
    rmse = 0
 
    for i in range(len(Y) + fc):
 
        if i == len(Y):
            Y.append((a[-1] + b[-1]) * s[-m])
 
        a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
        y.append((a[i + 1] + b[i + 1]) * s[i + 1])
 
    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
 
    return Y[-fc:], alpha, beta, gamma, rmse

def findDominentSeason(y, ignoreDC=True):
    '''
    Finds the dominant season in samples
    y: data
    ignoreDC: set to include DC component or not
    '''
    N = len(y)
    yf = sm.tsa.stattools.periodogram(y)
    xf = np.linspace(0,1,N/2) 

    strIndex = 0
    if ignoreDC:
        strIndex = 1
    
    ibest = np.argmax(yf[strIndex:N/2]) + strIndex

    return np.min([np.int_(np.round( 1.0/xf[ibest])), N])


def RMSE(params, *args):
    '''
    Root Mean Square Error function used to estimate the smoothing constants
    
    Parameters
    ----------
        params: alpha, beta, (gamma if type=additive)
        args[0]: Observations
        args[1]: type: 'linear' or 'additive' or 'multiplicative'
        args[2]: m: Number of seasons in data
        
    Returns
    ---------
    RMSE
    '''
    Y = args[0]
    type = args[1]
    rmse = 0
 
    if type == 'linear':
 
        alpha, beta = params
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        y = [a[0] + b[0]]
 
        for i in range(len(Y)):
 
            a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] + b[i + 1])
 
    else:
 
        alpha, beta, gamma = params
        m = args[2]        
        a = [sum(Y[0:m]) / float(m)]
        b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
 
        if type == 'additive':
 
            s = [Y[i] - a[0] for i in range(m)]
            y = [a[0] + b[0] + s[0]]
 
            for i in range(len(Y)):
 
                a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
                y.append(a[i + 1] + b[i + 1] + s[i + 1])
 
        elif type == 'multiplicative':
 
            s = [Y[i] / a[0] for i in range(m)]
            y = [(a[0] + b[0]) * s[0]]
 
            for i in range(len(Y)):
 
                a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
                y.append(a[i + 1] + b[i + 1] + s[i + 1])
 
        else:
 
            exit('Type must be either linear, additive or multiplicative')
        
    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))
 
    return rmse
   
