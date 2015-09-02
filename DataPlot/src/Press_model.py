'''
Created on 27 Nov 2014

@author: Manrich
'''
import dtw
from scipy import signal, stats

from Markov_model import Markov_model
import numpy as np
import tsutils


class Press_model(object):
    '''
    classdocs
    '''


    def __init__(self, data, corr_tresh=0.85, mean_ratio=0.1, match_rate=1.0, filter_window=21):
        '''
        Create a Signature Pattern recognizer Markov based on PRESS's signature-driven method
        
        Params
        ------
        @param data: Training data
        @type data: Array-like
        @param corr_tresh: Patterns should have a Pearson-Correlation above corr_tresh
        @type corr_tresh: float
        @param mean_ratio: 
        '''
        self.history = data[:]
        self.data = data[:]
        self.corr_thres = corr_tresh
        self.mr = mean_ratio
        self.window = filter_window
        self.match = match_rate
        self.N = data.shape[0]
        self.contain_pat = False
        self.warp = False
        self.markov = Markov_model(data=data)
        
    def fit(self):
        '''
        Fits a model: First smoothes the data using a Median filter. Then determines if a pattern exists
        by calculating the dominant period and splitting the time series into pattern windows.
        For each adjacent pair of the pattern widows, the Pearson-Correlation is calculated and compared 
        to the threshold together with the two pattern means. If the pattern exists for more than the match_ratio
        a Signature pattern is calculated using the original data and pattern windows that match.
        
        **Note: If no pattern is found, a 1st order Markov model is trained on the data instead
        '''
        self.markov.fit()
        # Smooth data and dominant period
        smooth = signal.medfilt(self.data, self.window)
        Z = tsutils.findDominentSeason(self.data)
#         Z=288
        # Determine the number of pattern windows and split smoothed data
        Q = np.int(np.ceil(1.0*self.N/Z))
        if Q > 2:
            idx = range(Z,Q*Z,Z)
            P = np.array(np.split(smooth, idx))
            self.P = P
            self.warp = P[-1].shape[0] != P[-2].shape[0]
            # Check pattern exist
            PC = []
            MR = []
            for i in range(Q-1-self.warp):
                PC.append(stats.pearsonr(P[i], P[i+1])[0])
                MR.append(np.mean(P[i])/np.mean(P[i+1]))
                
            # Test if pattern exist
            if np.all(PC>self.corr_thres):
                self.contain_pat = True
                Pr = np.array(np.split(self.data, idx))
                if self.warp:
                    self.sig = np.average(Pr[:-1],axis=0)
                else:
                    self.sig = np.average(Pr,axis=0)
        
    def update(self, newData):
        '''
        Updates the markov's training data and evaluates the pattern again. If a pattern does not exist
        call's Markov_models update() function with the new data.
        '''
        self.data = np.append(self.data[len(newData):], newData)
        self.contain_pat = False
        self.warp = False
        self.fit()
        self.markov.update(newData[:])
        self.history = np.append(self.history, newData)    
                    
    def predict(self, fc):
        '''
        Uses the Signature pattern and determines the index of the last pattern window, using Dynamic Time Warping
        and predicts fc samples into the future using the Signature pattern.
        
        **Note: Will return the Markov markov predictions if no pattern was found
        '''
        forecasts = np.zeros((fc))
        if self.contain_pat:
            if self.warp:
                Z = len(self.sig)
                index = getWarpIndex(self.P[-1], self.sig)
                if (index+fc)>=Z:
                    forecasts = np.array(self.sig[index:])
                    sub = (index+fc) - Z
                    forecasts = np.append(forecasts, self.sig[:sub])
                else:
                    forecasts = self.sig[index:fc+index]
            else:
                forecasts = self.sig[:fc]
        else:
            return self.markov.predict(fc)
        if forecasts.shape[0]<fc:
            f = np.zeros((fc))
            for i in range(forecasts.shape[0]):
                f[i] = forecasts[i]
            forecasts = f
        return forecasts
    
def getWarpIndex(x, y):
        dist, _, _= dtw.dtw(x, y, dist=absDist)
        return np.max(x.shape[0] + np.int(np.ceil(dist)-1),0)
    
def absDist(x, y):
    return abs(x-y)