'''
Created on 04 Feb 2015

@author: Manrich
'''
from __future__ import print_function

from __builtin__ import xrange
from multiprocessing import Pool 

from matplotlib import pyplot as plt
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FullConnection, IdentityConnection, BiasUnit
from pybrain.structure import RecurrentNetwork
from pybrain.structure.modules import LinearLayer, SigmoidLayer, LSTMLayer, TanhLayer
from pybrain.structure.modules import SigmoidLayer, TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.shortcuts import buildNetwork
import scipy
from sklearn.metrics import explained_variance_score
from statsmodels.tsa.vector_ar.var_model import forecast

import evaluation as eval
import fileutils
import numpy as np


def sampleGeometrically(A, B):
    if A<B:
        logA, logB = scipy.log(A), scipy.log(B)
    else:
        logA, logB = scipy.log(B), scipy.log(A)
    return scipy.exp(scipy.random.uniform(low=logA, high=logB))

def trainFunc(params):
    iter, trainds, validds, input_size, hidden, func, eta, lmda, epochs = params
    print('Iter:', iter, 'Epochs:', epochs, 'Hidden_size:', hidden, 'Eta:', eta, 'Lamda:', lmda, 'Activation:', func)
    
    # Build network
    n = RecurrentNetwork()
    n.addInputModule(LinearLayer(input_size, name = 'in'))
    n.addModule(func(hidden, name = 'hidden'))
    n.addModule(LinearLayer(hidden, name = 'context'))
    n.addOutputModule(LinearLayer(1, name = 'out'))
    n.addConnection(FullConnection(n['in'], n['hidden'], name = 'in_to_hidden'))
    n.addConnection(FullConnection(n['hidden'], n['out'], name = 'hidden_to_out'))
    n.addRecurrentConnection(FullConnection(n['hidden'], n['context']))
    rnet = n
    rnet.sortModules()
    
    trainer = BackpropTrainer(n, trainds, learningrate=eta, weightdecay=lmda, momentum=0.1, shuffle=False)
    trainer.trainEpochs(epochs)
    pred = np.nan_to_num(n.activateOnDataset(validds))
    validerr = eval.calc_RMSE(validds['target'], pred)
    varscore = explained_variance_score(validds['target'], pred)
    return validerr, varscore, n

if __name__ == '__main__':
    
    files =  fileutils.getFilelist("d:/data/cpu5")
#     target_files = ['cpu_1095481','cpu_1303745','cpu_1335782','cpu_1338948','cpu_1442486585','cpu_155313295','cpu_1664088958','cpu_317488701','cpu_317499484','cpu_3858945898','cpu_4304743890','cpu_4820238819','cpu_5796442','cpu_660404','cpu_711355','cpu_717319','cpu_904514','cpu_905062','cpu_907812']

    for machine in files[:]:   
#     for machine in target_files[16:19]:
        
        machine = machine.strip('.csv').split('/')[-1]
    #     machine = 'cpu_1095481'
        print(machine)
        
        data = np.genfromtxt("d:/data/cpu5/"+machine+".csv",skip_header=1, delimiter=',',usecols=(1))
        
        miniters=100
        maxiters=1000 
        TRAIN = 1000
        VALID = 100
        TEST = 100
        INPUT_SIZE = 30
        
        train, valid, test = data[:TRAIN], data[TRAIN-INPUT_SIZE:TRAIN+VALID], data[TRAIN-INPUT_SIZE+VALID:TRAIN+VALID+TEST]
        
        trainds = SupervisedDataSet(INPUT_SIZE, 1)
        testds = SupervisedDataSet(INPUT_SIZE, 1)
        validds = SupervisedDataSet(INPUT_SIZE, 1)
        
        for i in range(INPUT_SIZE,train.shape[0]):
            trainds.appendLinked(train[i-INPUT_SIZE:i],train[i])
        
        for i in range(INPUT_SIZE,test.shape[0]):
            testds.appendLinked(test[i-INPUT_SIZE:i],test[i])
        
        for i in range(INPUT_SIZE,valid.shape[0]):
            validds.appendLinked(valid[i-INPUT_SIZE:i],valid[i])
          
        THREADS = 4
        hidden_range=[4, 32]
        eta_range=[0.0001, 10.0]
        activation_func=[SigmoidLayer, TanhLayer]
        lamda_range=[1e-7, 1e-5]
        epochs_factor=1 
        
        besthparams = []
        besterr = np.inf
        bestvarscore = 0.0
        bestnet = None
        bestiter = maxiters
    
        pool = Pool(THREADS)
        for iter in range(1, maxiters+1, THREADS):
            hyperparams = []
            for i in range(THREADS):
                hidden = np.int(sampleGeometrically(hidden_range[0], hidden_range[1])) 
                eta = sampleGeometrically(eta_range[0], eta_range[1])
                lmda = 0.0
                if scipy.random.randint(0,2):
                    lmda = sampleGeometrically(lamda_range[0], lamda_range[1])
                func = activation_func[scipy.random.randint(low=0, high=len(activation_func))]
                epochs = 20
                hyperparams.append([iter+i, trainds, validds, INPUT_SIZE, hidden, func, eta, lmda, epochs])
            errors_var_net = pool.map(trainFunc, hyperparams)
            for i in range(THREADS):
                error = errors_var_net[i][0]
                varscore = errors_var_net[i][1]
                
                if varscore > bestvarscore:
                    if error < besterr:             
                        besterr = error
                        bestvarscore = varscore
                        besthparams = hyperparams[i][3:]
                        bestnet = errors_var_net[i][2]
                        bestiter = iter+i
            if iter>miniters and bestiter<iter/2:
                break 
            
        print('Epochs:', besthparams[-1], 'Hidden_size:', besthparams[1], 'Eta:', besthparams[3], 'Lamda:', besthparams[4], 'Activation:', besthparams[2])
        pool.close()
        pool.join() 
            
#         unknown = valid[-30:]
#         forecasts = []
#         for i in range(30):
#             fc = bestnet.activate(unknown)
#             forecasts.append(fc)
#             unknown = np.append(unknown[1:], fc)      
#         
#         plt.plot(validds['target'])
#         plt.plot(bestnet.activateOnDataset(validds))
#         plt.title('Validation')
#         plt.figure()
#         pred = bestnet.activateOnDataset(testds)
#         plt.plot(testds['target'])
#         plt.plot(pred)
#         plt.title('Testing')
#     #     plt.figure()
#     #     plt.plot(testds['target'][:30])
#     #     plt.plot(forecasts)
#     #     plt.title('Forecasts')
#         plt.show()
   
        NetworkWriter.writeToFile(bestnet, '../data/cpu5_rnn_networks/'+machine+".xml")
        with open('../data/cpu5_rnn_networks/hyperparams.csv', mode='a') as f:
            print([machine,besterr,besthparams[1],besthparams[3],besthparams[4],besthparams[2]], sep=',', end='\n', file=f)
    
    
