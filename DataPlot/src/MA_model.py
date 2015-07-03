'''
Created on 13 Nov 2014

@author: Manrich
'''
import numpy as np
import statsmodels.api as sm


class MA_model(object):
    '''
    Custom Moving Average model. Model parameters calculated using the average of previous values
    '''
    def __init__(self, data, order=30):
        '''
        Create a Moving Average model using the following equation: yt = sum(y_(t-i))/m for i in range of order m
        '''
        self.data = data[:]
        self.M = order
        
    def fit(self):
        '''
        No fitting required for MA model
        '''
        return
        
    def update(self, newData):
        '''
        Sets the training data of the model to data and refits the model 
        '''
        self.data = newData[:]
        
    def predict(self, fc):
        '''
        Predicts fc samples into the future using the Moving Average of m previous samples
        
        Params:
        ------
        @param fc: The number of samples to predict into the future, typically <= order of the model
        @type fc: int
        
        returns:
        ------
        @return: The fc predictions
        @rtype: array of shape (fc,1)  
        '''
        forecasts = []
        forecasts = np.zeros((fc,1))
        lookback = self.data[-self.M:]
        for i in range(fc):
            pred = np.sum(lookback)/self.M
            forecasts[i,0] = pred
            lookback = np.append(lookback[1:], pred)
        return forecasts
        