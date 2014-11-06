'''
# Holt-Winters algorithms to forecasting
# Coded in Python 2 by: Andre Queiroz
# Description: This module contains three exponential smoothing algorithms. They are Holt's linear trend method and Holt-Winters seasonal methods (additive and multiplicative).
# References:
#  Hyndman, R. J.; Athanasopoulos, G. (2013) Forecasting: principles and practice. http://otexts.com/fpp/. Accessed on 07/03/2013.
#  Byrd, R. H.; Lu, P.; Nocedal, J. A Limited Memory Algorithm for Bound Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5, pp. 1190-1208.

@author: Manrich
'''

from sys import exit
from math import sqrt
from numpy import array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import tsutils

class HW_model:
    '''
    Holt-winters model class
    '''

    def __init__(self, data, min, type):
        '''
        Create a Holt-Winters model of type={linear, additive, multiplicative}. The corresponding parameters [alpha, beta, gamma, m] 
        will be estimated from the data.
        
        Parameters
        ----------
        type: String specifying the type of method to use type={linear, additive, multiplicative}
        '''
        self.type = type
        self.alpha, self.beta, self.gamma = 0.0, 0.0, 0.0
        self.m = 0
        self.data = data[:]
        self.min = min
    
    def fit(self):
        '''
        Fit a Holt-winters method of type={linear, additive, multiplicative} on data
        
        Return
        ----------
        params: Array-like Parameters of the model [alpha, beta, gamma, m] depending on the type
        RMSE: Root Mean Squared Error of the model when fit to the data
        '''
        rmse = 0.0
        y = self.data[:]
        
        if self.type == 'linear':
            initial_values = array([0.3, 0.1])
            boundaries = [(0, 1), (0, 1)]
     
            parameters = fmin_l_bfgs_b(self.RMSE, x0 = initial_values, args = (y, self.type), bounds = boundaries, approx_grad = True)
            self.alpha, self.beta = parameters[0]
            rmse = parameters[1]
            return [self.alpha, self.beta], rmse
            
        elif self.type == 'additive':
            self.m = tsutils.findDominentSeason(y)
         
            initial_values = array([0.3, 0.1, 0.1])
            boundaries = [(0, 1), (0, 1), (0, 1)]
     
            parameters = fmin_l_bfgs_b(self.RMSE, x0 = initial_values, args = (y, self.type, self.m), bounds = boundaries, approx_grad = True)
            self.alpha, self.beta, self.gamma = parameters[0]
            rmse = parameters[1]
            
            return [self.alpha, self.beta, self.gamma, self.m], rmse
            
        elif self.type == 'multiplicative':
            self.m = tsutils.findDominentSeason(y)
            
            initial_values = array([0.0, 1.0, 0.0])
            boundaries = [(0, 1), (0, 1), (0, 1)]
     
            parameters = fmin_l_bfgs_b(self.RMSE, x0 = initial_values, args = (y, self.type, self.m), bounds = boundaries, approx_grad = True)
            self.alpha, self.beta, self.gamma = parameters[0]
            rmse = parameters[1]
            
            return [self.alpha, self.beta, self.gamma, self.m], rmse 

    def update(self,data):
        '''
        Updates the Holt-winters model by re-estimating the paramters {alpha, beta, gamma, m} depending
        on the model type
        '''
        self.data = data[:]
        y = self.data[:]
        if self.type == 'linear':
            initial_values = array([self.alpha, self.beta])
            boundaries = [(0, 1), (0, 1)]
     
            parameters = fmin_l_bfgs_b(self.RMSE, x0 = initial_values, args = (y, self.type), bounds = boundaries, approx_grad = True)
            self.alpha, self.beta = parameters[0]
            rmse = parameters[1]
            return [self.alpha, self.beta], rmse
            
        elif self.type == 'additive':
            self.m = tsutils.findDominentSeason(y)
         
            initial_values = array([self.alpha, self.beta, self.gamma])
            boundaries = [(0, 1), (0, 1), (0, 1)]
     
            parameters = fmin_l_bfgs_b(self.RMSE, x0 = initial_values, args = (y, self.type, self.m), bounds = boundaries, approx_grad = True)
            self.alpha, self.beta, self.gamma = parameters[0]
            rmse = parameters[1]
            
            return [self.alpha, self.beta, self.gamma, self.m], rmse
            
        elif self.type == 'multiplicative':
            self.m = tsutils.findDominentSeason(y)
            
            initial_values = array([self.alpha, self.beta, self.gamma])
            boundaries = [(0, 1), (0, 1), (0, 1)]
     
            parameters = fmin_l_bfgs_b(self.RMSE, x0 = initial_values, args = (y, self.type, self.m), bounds = boundaries, approx_grad = True)
            self.alpha, self.beta, self.gamma = parameters[0]
            rmse = parameters[1]
        
    def predict(self, fc):
        '''
        Forecasts fc samples into the future, using the model parameters corresponding to the model type
        
        Parameters
        ----------
        fc: Number of samples ahead to predict
        
        Return
        ----------
        Forecast: List of predicted values of length fc
        '''
        Y = self.data[:]
        if self.type == 'linear':
            alpha, beta = self.alpha, self.beta
            a = [Y[0]]
            b = [Y[1] - Y[0]]
            y = [a[0] + b[0]]
         
            for i in range(len(Y) + fc):
                if i == len(Y):
                    Y.append(a[-1] + b[-1])
         
                a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                y.append(a[i + 1] + b[i + 1])
                
        elif self.type == 'additive':
            alpha, beta, gamma, m = self.alpha, self.beta , self.gamma, self.m
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
                
        elif self.type == 'multiplicative':
            alpha, beta, gamma, m = self.alpha, self.beta , self.gamma, self.m
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
                
        ''' Replace negative values'''
        predictions = np.array(Y[-fc:])
        predictions[np.argwhere(predictions<0)] = np.float_(self.min)
        return predictions
   
    def RMSE(self,params, *args):
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