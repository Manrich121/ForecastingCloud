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


class Entwine_model(object):
    '''
    classdocs
    '''

    def __init__(self, data, machineID, netPath, eta, lmda, input_size=30, epochs=20, train_str_index=1000, train_end_index=3000):
        '''
        Constructor
        '''
        self.cpuData = data[0]
        self.memData = data[1]
        self.machineID = machineID
        self.eta = eta
        self.lmda = lmda
        self.INPUT_SIZE = input_size
        self.epochs = epochs
        self.str_train = train_str_index
        self.end_train = train_end_index
        self.net = NetworkReader.readFrom(netPath)
        
        self.memForecasts = np.genfromtxt("d:/data/memory_fnn/"+machineID.replace("cpu", "memory"),delimiter=',').ravel()
        
    def fit(self):
        trainds = SupervisedDataSet(2*self.INPUT_SIZE, 1)
        for i in range(self.str_train, self.end_train):
            trainds.appendLinked(np.append(self.cpuData[i-self.INPUT_SIZE:i], self.memData[i-self.INPUT_SIZE:i]).ravel(), self.cpuData[i])
                
        trainer = BackpropTrainer(self.net, trainds, learningrate=self.eta, weightdecay=self.lmda, momentum=0.1, shuffle=False)
        trainer.trainEpochs(self.epochs)
                    
        trainer = None
        
    def update(self, data):
        # Increment training indexes 
        self.str_train = self.end_train
        self.end_train += self.INPUT_SIZE
        if self.end_train > len(self.cpuData):
            self.end_train = len(self.cpuData)
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
        unknown= np.append(self.cpuData[self.end_train-self.INPUT_SIZE:self.end_train], self.memData[self.end_train-self.INPUT_SIZE:self.end_train]).ravel()
        forecasts = []
        for i in range(self.INPUT_SIZE):
            fc = self.net.activate(unknown)
            forecasts.append(fc)
            unknownCpu = np.append(self.cpuData[self.end_train+i+1-self.INPUT_SIZE:self.end_train], forecasts)
            unknownMem = np.append(self.memData[self.end_train+i+1-self.INPUT_SIZE:self.end_train], self.memForecasts[:i+1])
            unknown = np.append(unknownCpu, unknownMem).ravel()
            
        return forecasts