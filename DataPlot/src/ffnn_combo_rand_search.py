'''
Created on 04 Feb 2015

@author: Manrich
'''
from __future__ import print_function
from multiprocessing import Pool 

from matplotlib import pyplot as plt
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import SigmoidLayer, TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.shortcuts import buildNetwork
import scipy
from sklearn.metrics import explained_variance_score

import evaluation as eval
import fileutils
import numpy as np

from pybrain.structure import FeedForwardNetwork, FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer

def sampleGeometrically(A, B):
    if A<B:
        logA, logB = scipy.log(A), scipy.log(B)
    else:
        logA, logB = scipy.log(B), scipy.log(A)
    return scipy.exp(scipy.random.uniform(low=logA, high=logB))

def buildNet(input_size, hidden_size):
    n = FeedForwardNetwork()
    in1Layer = LinearLayer(input_size)
    in2Layer = LinearLayer(input_size)
    hidden1Layer = SigmoidLayer(hidden_size)
    hidden2Layer = SigmoidLayer(hidden_size)
    hidden3Layer = SigmoidLayer(2)
    outLayer = LinearLayer(1)
    
    n.addInputModule(in1Layer)
    n.addInputModule(in2Layer)
    n.addModule(hidden1Layer)
    n.addModule(hidden2Layer)
    n.addModule(hidden3Layer)
    n.addOutputModule(outLayer)
    
    in1_to_hidden1 = FullConnection(in1Layer, hidden1Layer)
    in2_to_hidden2 = FullConnection(in2Layer, hidden2Layer)
    hidden1_to_hidden3 = FullConnection(hidden1Layer, hidden3Layer)
    hidden2_to_hidden3 = FullConnection(hidden2Layer, hidden3Layer)
    hidden3_to_out = FullConnection(hidden3Layer, outLayer)
    
    n.addConnection(in1_to_hidden1)
    n.addConnection(in2_to_hidden2)
    n.addConnection(hidden1_to_hidden3)
    n.addConnection(hidden2_to_hidden3)
    n.addConnection(hidden3_to_out)
    n.sortModules()
    
    return n

def trainFunc(params):
    iter, trainds, validds, input_size, hidden, func, eta, lmda, epochs = params
    print('Iter:', iter, 'Epochs:', epochs, 'Hidden_size:', hidden, 'Eta:', eta, 'Lamda:', lmda, 'Activation:', func)
    net = buildNet(input_size, hidden)
    trainer = BackpropTrainer(net, trainds, learningrate=eta, weightdecay=lmda, momentum=0.1, shuffle=False)
    trainer.trainEpochs(epochs)
    pred = np.nan_to_num(net.activateOnDataset(validds))
    validerr = eval.calc_RMSE(validds['target'], pred)
    varscore = explained_variance_score(validds['target'], pred)
    return validerr, varscore, net

if __name__ == '__main__':
    
    files =  fileutils.getFilelist("d:/data/cpu")
    for machine in files[14:16]:
        curMachine = machine.strip('.csv').split('/')[-1]
        print(curMachine)
        cpuData = np.genfromtxt("d:/data/cpu/"+curMachine+".csv",delimiter=',',skip_header=1,usecols=(1))
        memData = np.genfromtxt("d:/data/memory/"+curMachine.replace('cpu','memory')+".csv",delimiter=',',skip_header=1,usecols=(1))
        
        TRAIN = 1000
        VALID = 100
        TEST = 100
        INPUT_SIZE = 30
        
        cputrain, cpuvalid, cputest = cpuData[:TRAIN], cpuData[TRAIN-INPUT_SIZE:TRAIN+VALID], cpuData[TRAIN-INPUT_SIZE+VALID:TRAIN+VALID+TEST]
        memtrain, memvalid, memtest = memData[:TRAIN], memData[TRAIN-INPUT_SIZE:TRAIN+VALID], memData[TRAIN-INPUT_SIZE+VALID:TRAIN+VALID+TEST]
        
        trainds = SupervisedDataSet(2*INPUT_SIZE, 1)
        testds = SupervisedDataSet(2*INPUT_SIZE, 1)
        validds = SupervisedDataSet(2*INPUT_SIZE, 1)
        
        for i in range(INPUT_SIZE, TRAIN):
            trainds.appendLinked(np.append(cputrain[i-INPUT_SIZE:i], cputrain[i-INPUT_SIZE:i]).ravel(),cputrain[i])
        
        for i in range(INPUT_SIZE,cputest.shape[0]):
            testds.appendLinked(np.append(cputest[i-INPUT_SIZE:i], memtest[i-INPUT_SIZE:i]).ravel(),cputest[i])
        
        for i in range(INPUT_SIZE,cpuvalid.shape[0]):
            validds.appendLinked(np.append(cpuvalid[i-INPUT_SIZE:i], memvalid[i-INPUT_SIZE:i]).ravel(),cpuvalid[i])
          
        THREADS = 4
        hidden_range=[4, 32]
        eta_range=[0.0001, 10.0]
        activation_func=[SigmoidLayer, TanhLayer]
        lamda_range=[1e-7, 1e-5]
        epochs_factor=1
        miniters=100
        maxiters=1000  
        
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
            
#         unknowncpu = cpuvalid[-30:]
#         unknownmem = memvalid[-30:]
#         forecasts = []
#         for i in range(30):
#             fc = bestnet.activate(np.append(cpuvalid, memvalid).ravel())
#             forecasts.append(fc)
#             unknowncpu = np.append(unknowncpu[1:], fc)      
         
#         plt.plot(validds['target'])
#         plt.plot(bestnet.activateOnDataset(validds))
#         plt.title('Validation')
#         plt.figure()
#         pred = bestnet.activateOnDataset(testds)
#         plt.plot(testds['target'])
#         plt.plot(pred)
#         plt.title('Testing')
#         plt.show()
#     
        NetworkWriter.writeToFile(bestnet, '../data/cpu_mem_combonet/'+curMachine+".xml")
        with open('../data/cpu_mem_combonet/hyperparams.csv', mode='a') as f:
            print([machine,besterr,besthparams[1],besthparams[3],besthparams[4],besthparams[2]], sep=',', end='\n', file=f)
    
    
