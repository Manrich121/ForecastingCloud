'''
Created on 13 Nov 2014

@author: Manrich
'''
import numpy as np
import statsmodels.api as sm

class AR_model(object):
    '''
    Custom Autoregressive model. Model parameters calculated using the autocorrelation coefficients
    '''
    def __init__(self, data, order=30):
        '''
        Create a Autoregressive model using the following equation: Xt = mu + a1*(Xt-1 - mu) + a2*(Xt-2 - mu) + ...
        where mu is the mean of the data and a1-az are determined by solving linear equations:
        
        sum_i=1_to_Z( ai*R(i-j) ) = R(j) for 1<=j<=Z, where R(k) is the kth auto-correlation coefficient of the
        data and Z is the order of the model.
        '''
        self.data = data[:]
        self.Z = order
        
    def fit(self):
        '''
        Fits the model using the data and calculating the parameters by solving the linear equations:
        sum_i=1_to_Z( ai*R(i-j) ) = R(j) for 1<=j<=Z, where R(k) is the kth auto-correlation coefficient of the
        data and Z is the order of the model.
        
        Returns:
        -------
        @return: the parameters of the model a1-az
        @rtype: array like of shape (Z,1)
        '''
        Z = self.Z
        R = sm.tsa.stattools.acf(self.data, nlags=Z)
        R = np.r_[R,R[::-1]][:-1]
        
#         Setting up model params
        a = np.zeros((Z,Z))
        b = np.zeros((Z,1))
        for j in range(1,Z+1):
            b[j-1] = R[j]
            for i in range(1,Z+1):
                a[j-1,i-1] = R[i-j]
                
        self.params = np.linalg.solve(a, b)
        
    def update(self, newData):
        '''
        Sets the training data of the model to data and refits the model 
        '''
        self.data = newData[:]
        self.fit()
        
    def predict(self, fc):
        '''
        Predicts fc samples into the future using the model params and training data
        
        Params:
        ------
        @param fc: The number of samples to predict into the future, typically <= order of the model
        @type fc: int
        
        returns:
        ------
        @return: The fc predictions
        @rtype: array of shape (fc,1)  
        '''
        window = fc
        mu = np.mean(self.data)
        forecasts = np.zeros((window,1))
        x = np.zeros((1,self.Z))
        for f in range(window):
            for i in range(1,f+1):
                x[0,i-1] = forecasts[f-i,0]
            for t in range(self.Z-f):
                x[0,t+f] = self.data[-(t+1)]
            forecasts[f,0] = np.dot(x-mu, self.params) + mu  
#             for t in range(self.Z-f):
#                 x[0,t] = self.data[-(t+1)]
#             if f>0:
#                 x[0,-f] = forecasts[f-1,0]
#             forecasts[f,0] = np.dot(x-mu, self.params) + mu
             
        return forecasts
        