'''
Created on 13 Nov 2014

@author: Manrich
'''
import numpy as np


class Norm_model(object):
    '''
    Normal (Gaussian) distribution Model. 
    '''

    def __init__(self, data):
        '''
        Create a normal distribution model
        '''
        self.data = data[:]
        
    def fit(self):
        '''
        Fits a Gaussian model on the training data by calculating parameters mu and sigma
        '''
        self.mu = np.mean(self.data)
        self.sigma = np.std(self.data)
    
    def update(self, data):
        '''
        Updates the model training data and refits the Gaussian model
        '''
        self.data = data
        self.fit()
    
    def predict(self, fc):
        '''
        Samples the Normal distribution to obtain fc number of samples
        '''
        return np.random.normal(self.mu, self.sigma, fc)