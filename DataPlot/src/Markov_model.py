'''
Created on 17 Nov 2014

@author: Manrich
'''
import numpy as np
import pykov

class Markov_model(object):
    '''
    classdocs
    '''
    def __init__(self, data, num_states=40):
        '''
        Constructor
        '''
        self.data = data[:]        
        self.bins = np.array(np.linspace(0,max(data), num_states+1))
        
        
    def fit(self):
        transmat = pykov.Chain()
        for i in range(len(self.data)-1):
            t = pykov.Chain({(self.getState(self.data[i]), self.getState(self.data[i+1])):1})
            transmat = transmat + t
        transmat.stochastic()
        self.transmat = pykov.Chain(transmat)
    
    def update(self,data):
        self.data = data[:]
        self.fit()
        
#     def predict(self, fc):
#         cur_state = self.getState(self.data[-1])
#         walk_states = self.transmat.walk(fc, start=cur_state)
#         return np.atleast_2d(self.bins[walk_states]).T
    
    def predict(self, fc):
        forecasts = np.zeros((fc,1))
        cur_state = pykov.Vector({self.getState(self.data[-1]):1})
        for i in range(1,fc+1):
            transrow = self.transmat.pow(cur_state, i)
            pred_state = transrow.keys()[np.argmax(transrow.values())]
            forecasts[i-1,0] = self.bins[pred_state]
        return forecasts
     
    def getState(self,val):
        return np.argmin(val>self.bins)