'''
Created on 17 Nov 2014

@author: Manrich
'''
import numpy as np

class Markov_model(object):
    '''
    classdocs
    '''
    def __init__(self, data, maximum, M=40, order=1):
        '''
        Constructor
        '''
        self.data = data[:]        
        self.bins = np.array(np.linspace(min(data),maximum, M))
        self.states = np.digitize(data, self.bins, right=True)
        self.M = M
        self.order = order
        if order == 1:
            self.transcount = np.ones((self.M,self.M))
        elif order == 2:
            self.transcount = np.array([np.zeros((M,M))]*M)
           
    def fit(self):
        '''
        Initialises the transition matrix using the frequency of occurance and Laplace method
        '''
        if self.order == 1:
            for n in range(self.data.shape[0]-1):
                i = self.states[n]-1
                j = self.states[n+1]-1
                self.transcount[i, j] += 1
            self.transmat = np.nan_to_num(self.transcount / np.tile(np.sum(self.transcount, axis=1), (self.M,1)).T)
        elif self.order == 2:
            M = self.M
            for i in range(1,len(self.states)-1):
                x, y, z = self.states[i-1], self.states[i], self.states[i+1]
                self.transcount[x-1,y-1,z-1] += 1
            self.transmat = np.nan_to_num(np.divide(self.transcount, np.array([np.tile(np.sum(self.transcount[idx], axis=1), (M,1)).T for idx in range(M)])))

    def update(self, newData):
        self.data = newData[:]
        self.bins = np.array(np.linspace(min(newData),max(newData), self.M))
        self.states = np.digitize(newData, self.bins, right=True)
        self.fit()
        
    
    def predict(self, fc):
        forecasts = np.zeros((fc))
        str_state = self.states[-1] 
        if self.order == 1:
            for p in range(1,fc+1):
                transrow = self.transmat[str_state-1,:]
                pred_state = np.argmax(transrow)+1
                str_state = pred_state
                forecasts[p-1] = self.bins[pred_state]
#             str_state = np.zeros((1, self.M))
#             str_state[0, self.states[-1]-1] = 1  
#             trans = np.copy(self.transmat)
#             for p in range(fc):
#                 transrow =  np.dot(str_state, trans)
#                 forecasts[p] = self.bins[np.argmax(transrow) +1]
#                 trans = np.dot(self.transmat, trans)
        elif self.order == 2:
            prev_state = self.states[-2]
            for p in range(fc):
                pred = np.argmax(self.transmat[prev_state-1, str_state-1]) +1
                prev_state = str_state
                str_state = pred
                forecasts[p] = self.bins[pred]
        return forecasts