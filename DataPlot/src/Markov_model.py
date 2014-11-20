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
    def __init__(self, data, M=40):
        '''
        Constructor
        '''
        self.data = data[:]        
        self.bins = np.array(np.linspace(0,max(data), M+1))
        self.M = M
          
    def fit(self):
        '''
        Initialises the transition matrix using the fequency of occurance and Laplace method
        '''
#         transmat = np.ones((self.M,self.M))
        transmat = np.zeros((self.M,self.M))
        for n in range(self.data.shape[0]-1):
            i, j = self.getState(self.data[n])-1 , self.getState(self.data[n+1])-1
            transmat[i, j] += 1
        transmat = transmat / np.tile(np.sum(transmat,axis=1), (self.M,1)).T
        self.transmat =  transmat
            
    def update(self,data):
        self.data = data[:]
        self.bins = np.array(np.linspace(0,max(data), self.M+1))
        self.fit()
    
    def predict(self, fc):
        forecasts = np.zeros((fc,1))
        transback = self.transmat
        for p in range(1,fc+1):
            cur_state = np.zeros((1,self.M))
            cur_state[0, self.getState(self.data[-1])-1] = 1 
            transrow =  np.dot(cur_state, self.transmat**p).squeeze()
            pred_state = np.argmax(transrow)+1
            forecasts[p-1,0] = self.bins[pred_state]
            self.transmat = np.dot(self.transmat, self.transmat) 
        self.transmat = transback
        return forecasts
     
    def getState(self,val):
        return np.argmin(val>self.bins)