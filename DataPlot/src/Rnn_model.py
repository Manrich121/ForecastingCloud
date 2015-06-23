'''
Created on 16 Feb 2015

@author: Manrich
'''
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.tools.shortcuts import buildNetwork
import scipy

import evaluation as eval
import numpy as np


class Rnn_model(object):
    '''
    classdocs
    '''

    def __init__(self, data, machineID, eta, lmda, netPath, input_size=30, epochs=20, train_str_index=1000, train_end_index=3000):
        '''
        Constructor
        '''
        self.data = data
        self.machineID = machineID
        self.eta = eta
        self.lmda = lmda
        self.INPUT_SIZE = input_size
        self.epochs = epochs
        self.str_train = train_str_index
        self.end_train = train_end_index
        self.net = NetworkReader.readFrom(netPath)

        
    def fit(self):
        trainds = SupervisedDataSet(self.INPUT_SIZE, 1)
        for i in range(self.str_train, self.end_train):
            trainds.appendLinked(self.data[i-self.INPUT_SIZE:i],self.data[i])
        
        trainer = BackpropTrainer(self.net, trainds, learningrate=self.eta, weightdecay=self.lmda, momentum=0.1, shuffle=False)
        trainer.trainEpochs(self.epochs)
                    
        trainer = None
        
    def update(self, data):
        # Increment training indexes 
        self.str_train = self.end_train
        self.end_train += self.INPUT_SIZE
        if self.end_train > len(self.data):
            self.end_train = len(self.data)
        self.fit()

    def predict(self, fc):
        '''
        Predicts fc samples into the future using the network corresponding to the data file
        
        Params:
        ------
        @param fc: The number of samples to predict into the future, typically <= order of the model
        @type fc: int
        
        returns:
        ------
        @return: The fc predictions
        @rtype: array of shape (fc,)  
        '''
        unknown = self.data[self.end_train-self.INPUT_SIZE:self.end_train]
        forecasts = []
        for i in range(self.INPUT_SIZE):
            fc = self.net.activate(unknown)
            forecasts.append(fc)
            unknown = np.append(unknown[1:], fc)
            
        return np.array(forecasts)