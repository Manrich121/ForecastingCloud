'''
Created on 03 Dec 2014

@author: Manrich
'''
import numpy as np
import pywt
from AR_model import AR_model

class Wavelet_model(object):
    '''
    A Auto-regressive model that utilises wavelets to decompose the data into multiple wavelets
    '''

    
    def __init__(self, data, levels=5, types = ['db1', 'db2', 'bior1.5', 'sym4'], order = [1, 1, 2, 4, 8, 16]):
        '''
        Creates a Wavelet-based Auto-regression model.
        
        Params
        ------
        @param data: Training data transformed using wavelets and building AR models.
        @param levels: The number of levels used in the wavelet transform / decomposition.
        @param type: A list of wavelet types a present in PyWavelets package. Length must match levels.
        @param order: The order of AR(p) model used for the approximated signal and  n detail signals. Length must be levels+1
        '''
        
        self.data = data[:]
        self.levels= levels
        self.types = types
        self.order = order
        self.models = []
        
    def fit(self):
        '''
        First finds the wavelet in types that fits the data the best => The smallest Euclidean distance between the 
        approximation signal and under-sampled signal
        After finding the best Wavelet, uses it to decompose data into levels detail signals and builds a AR(p)
        model per approximation signal and detail signals using order as p for each corresponding  
        '''
        self.bestType = 'db1'
        bestDist = np.inf

        for t in self.types:
            approx_levels = pywt.wavedec(self.data, wavelet=t, level=self.levels)
            idx = np.int_(np.linspace(0, self.data.shape[0]-1, num=len(approx_levels[0])))
            samples = self.data[idx]
            dist = np.linalg.norm(approx_levels[0]-samples)
            if dist < bestDist:
                bestDist = dist
                self.bestType = t
                
        self.coefs = pywt.wavedec(self.data, wavelet=self.bestType, level=self.levels)
        
        for i in range(len(self.order)):
            model = AR_model(approx_levels[i], order=self.order[i])
            model.fit()
            self.models.append(model)
        
    def update(self, data):
        '''
        Replaces data, finds best wavelet function and refits AR models
        '''
        self.data = data[:]
        self.models = []
        self.fit()    

        
    def predict(self, fc):
        '''
        Predicts fc samples into the future, by predicting on each AR model and inverse Wavelet transform to get
        the predictions.
        
        **Note fc <= levels
        '''
        coefs = self.coefs
        for i in range(len(self.order)):
 
            pred = self.models[i].predict(self.order[i])
            coefs[i] = np.append(coefs[i], pred)
        index = np.power(2 , self.levels)
        forecasts = pywt.waverec(coefs, wavelet=self.bestType)[-index:-(index - fc)]
        return forecasts
        