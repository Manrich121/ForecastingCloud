'''
Created on 01 Jun 2015

@author: Manrich
'''

import StringIO
from multiprocessing import Pool as ThreadPool 
import sys, os

import MA_model
import AR_model
import HW_model
import Markov_model
import Press_model
import Wavelet_model
import Fnn_model
import Rnn_model
import Entwine_model
import evaluation as eval
import fileutils
import numpy as np

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer, LinearLayer

# Main definition - constants
menu_actions  = {}  
methods_dict = {}
# INPUT = 'd:/data/'
# OUTPUT = 'd:/data/'
# OUTPUT = 'D:/15_sample_results/'

# INPUT = 'd:/Wikipage data/'
# OUTPUT = 'd:/Wikipage data/'

INPUT = 'D:/7h_data/'
OUTPUT= 'D:/7h_data/'

TYPE = None
METHOD = None


def performsSlidingWindowForecast(params, minpercentile=5, order=16, training_window=30, input_window=32, predic_window=1):
    '''
    Input window = 250 hours = 250*12 = 3000 
    look ahead window 60 samples =  5 hours = 720min/5 = 60
    
    '''
    filename, METHOD, TYPE, OUTPUT = params[0:4]
#Wikidata
#     data = np.genfromtxt(filename)
#     data = data/np.max(data)
    lastFc = None
    if TYPE == 'pageviews' or TYPE == 'network':
        data = np.nan_to_num(np.genfromtxt(filename.replace(".csv",""))).ravel()
        data = data/np.max(data)
    elif OUTPUT == 'D:/7h_data/':
        data = np.nan_to_num(np.genfromtxt(filename, dtype=float)).ravel()
    else:
        data = np.nan_to_num(np.genfromtxt(filename, delimiter=',', skip_header=1)[:,1]).ravel()
    minimum = np.percentile(data,minpercentile)
    N = len(data)
    result = []
    print filename, "started..."
    for strIndex in range(0,N-input_window - predic_window, predic_window):
        if strIndex == 0:
            y = data[:input_window]
            if METHOD == 'ar2':
                model = AR_model.AR_model(y, order=order)
            elif METHOD == 'ma':
                model = MA_model.MA_model(y,order=30)
            elif METHOD == 'hw':
                model = HW_model.HW_model(y, minimum, 'additive')
            elif METHOD == 'markov1':
                model = Markov_model.Markov_model(y, order=1)
            elif METHOD == 'markov2':
                model = Markov_model.Markov_model(y, order=2)
            elif METHOD == 'press':
                model = Press_model.Press_model(y)
            elif METHOD == 'agile':
                model = Wavelet_model.Wavelet_model(y)
            elif METHOD == 'fnn':
                filename, METHOD, TYPE, OUTPUT, INPUT, curEta, curLmda = params[:7]
                curMachine = filename.split('/')[-1].replace(".csv",".xml")
                if TYPE.startswith("memory"):
                    curMachine = curMachine.replace("memory", "cpu")
                    model = Fnn_model.Fnn_model(data=data, machineID = curMachine, netPath="../data/"+TYPE.replace("memory", "cpu")+"_networks/"+curMachine, eta=curEta, lmda=curLmda)
                else:
                    model = Fnn_model.Fnn_model(data=data, machineID = curMachine, netPath="../data/"+TYPE+"_networks/"+curMachine, eta=curEta, lmda=curLmda)
            elif METHOD == 'rnn':
                filename, METHOD, TYPE, OUTPUT, INPUT, curEta, curLmda = params[:7]
                curMachine = filename.split('/')[-1]
                if TYPE.startswith("memory"):
                    curMachine = curMachine.replace("memory", "cpu")
                    model = Rnn_model.Rnn_model(data=data, machineID = curMachine, netPath="../data/"+TYPE.replace("memory", "cpu")+"_networks/"+curMachine.replace(".csv",".xml"), eta=curEta, lmda=curLmda)
                else:
                    model = Rnn_model.Rnn_model(data=data, machineID = curMachine, netPath="../data/"+TYPE+"_rnn_networks/"+curMachine.replace(".csv",".xml"), eta=curEta, lmda=curLmda)
            elif METHOD == 'entwine':
                filename, METHOD, TYPE, OUTPUT, INPUT, curEta, curLmda = params[:7]
                curMachine = filename.split('/')[-1]
                data2 = np.nan_to_num(np.genfromtxt(filename.replace("cpu", "memory"), delimiter=',', skip_header=1)[:,1]).ravel()
                
                model = Entwine_model.Entwine_model([data, data2], machineID = curMachine, netPath="../data/entwine_networks/"+curMachine.replace(".csv",".xml"), eta=curEta, lmda=curLmda)
                
            model.fit()
        else:
            if METHOD == 'press':
                y = data[input_window + strIndex - predic_window:input_window + strIndex]
            else:
                y = data[strIndex:strIndex+input_window]
            model.update(y)
        
        p = model.predict(predic_window)
        y_pred = np.atleast_2d(p)
        y_pred = np.reshape(y_pred, (predic_window,1)) 
        
        if METHOD == 'rnn':
            if lastFc is not None:
                y_pred[0,0] = lastFc 
            lastFc = y_pred[-1,0]

        y_pred[y_pred[:,0]<0,0] = minimum
        
        result.append(y_pred[:,0])
    f = filename.split('/')[-1]
    fileutils.writeCSV(OUTPUT+TYPE+"_"+METHOD+"/"+f, np.atleast_2d(result))
    print filename, "complete!"
     
def ensembleModel(params, types=['ma','ar','fnn','agile'], step=30, input_window=3000):
    input_size = len(types)
    filename, METHOD, TYPE, OUTPUT = params[0:4]
    filename = filename.split('/')[-1]
    
    filename, METHOD, TYPE, OUTPUT = params[0:4]
    filename = filename.split('/')[-1]    
    
    combine_model = np.genfromtxt(OUTPUT+TYPE+"_"+types[0]+"/"+filename, delimiter=',', usecols=range(0,30)).ravel()
    truevals = np.nan_to_num(np.genfromtxt(OUTPUT+TYPE+"/"+filename, delimiter=',',skip_header=1, usecols=(1))[:input_window+len(combine_model)])
        
    for t in types[1:]:
        forecasts = np.genfromtxt(OUTPUT+TYPE+"_"+t+"/"+filename, delimiter=',', usecols=range(0,30)).ravel()
        combine_model = np.vstack((combine_model, forecasts))
        
    average_fc = np.average(combine_model, axis=0)
    
    if METHOD == 'avg4':
        fileutils.writeCSV(OUTPUT+TYPE+"_"+METHOD+"/"+filename, np.atleast_2d(average_fc).reshape([178,30]))
        print filename, "complete"
        return
    if METHOD == 'combo4' or METHOD =='wa':
        training = SupervisedDataSet(input_size, 1)
        for i in range(step):
            
            training.appendLinked([combine_model[t][i] for t in range(input_size)], truevals[i+input_window])
            
        besterr = eval.calc_RMSE(truevals[input_window:input_window+step], average_fc[:step])
        bestNet = None
        
        for i in range(50):
            if METHOD == 'wa':
                net = buildNetwork(input_size, 1, hiddenclass=LinearLayer, bias=False)
            else:
                net = buildNetwork(input_size, 2, 1, hiddenclass=LinearLayer, bias=False)
            trainer = BackpropTrainer(net, training, learningrate=0.001, shuffle=False)
            trainer.trainEpochs(100)
    
            err = eval.calc_RMSE(truevals[input_window:input_window+step], net.activateOnDataset(training))
            if err < besterr:
                bestNet = net
                break
            
        combo_fc = average_fc[0:step].tolist()
#         combo_fc = []
        if bestNet == None:
            combo_fc = average_fc
        else:
            for i in range(step, len(combine_model[0]), step):
                training = SupervisedDataSet(input_size, 1)
                for j in range(i,i+step):
                    combo_fc.append(bestNet.activate([combine_model[t][j] for t in range(input_size)])[0])
                    training.appendLinked([combine_model[t][j] for t in range(input_size)], truevals[j+input_window])
                trainer = BackpropTrainer(bestNet, training, learningrate=0.01, shuffle=False)
                trainer.trainEpochs(2)
                
        result  = np.atleast_2d(combo_fc).reshape([178,30])
        minimum = np.percentile(truevals,5)
        result[0,result[0,:] < minimum] = minimum 
        
        fileutils.writeCSV(OUTPUT+TYPE+"_"+METHOD+"/"+filename, result)
        print filename, "complete"
    
    
def performEvaluations(params, train_window = 3000, overload_dur = 5, overload_percentile = 70, predic_window=30):
    
    filename, METHOD, TYPE, OUTPUT, INPUT = params[:5]
    filename = filename.split('/')[-1]
    print OUTPUT+TYPE+"_"+METHOD+"/" + filename, "started..."
    
    cur_results = []
    forecasts = np.nan_to_num(np.genfromtxt(OUTPUT+TYPE+"_"+METHOD+"/" + filename, delimiter=',',usecols=range(0,predic_window))).ravel() # ,usecols=range(0,30)
    
    if TYPE == 'pageviews' or TYPE == 'network':
        filename = filename.replace(".csv","")
        truevals = np.genfromtxt(INPUT+TYPE+"/"+filename)[:train_window+len(forecasts)]
        truevals = truevals/np.max(truevals)
    else:
        truevals = np.genfromtxt(INPUT+TYPE+"/"+filename, delimiter=',',skip_header=1)[:train_window+len(forecasts),1]
    
    # Normalize
#     truevals = np.divide(truevals, np.max(truevals))
    
    threshold =  np.percentile(truevals, overload_percentile)
    
    cur_results.append(eval.calc_RMSE(truevals[train_window:], forecasts))
    for val in eval.calc_upper_lower_acc(truevals[train_window:], forecasts):
        cur_results.append(val) 
    for val in eval.calc_persample_accuracy(truevals[train_window:], forecasts, threshold):
        cur_results.append(val)
    for val in eval.calc_overload_states_acc(truevals[train_window:], forecasts, threshold):
        cur_results.append(val)
        
    return cur_results
 
# Exit program
def exit():
    print "Goodbye :)"
    sys.exit()
 
methods_dict = {
    '1': 'hw',
    '2': 'ar2',
    '3': 'markov1',
    '4': 'markov2',
    '5': 'press',
    '6': 'agile',
    '7': 'fnn',  
    '8': 'rnn', 
    '9': 'entwine', 
    '10': 'ma',
    '11': 'avg4',
    '12': 'combo4',
    '13': 'wa',     
}
 
# =======================
#      MAIN PROGRAM
# =======================

def main():
    global METHOD
    global TYPE
    global OUTPUT
    global INPUT
    # Launch main menu
#     main_menu()
    print "Comparing Forecasting methods:\n"
    print "Please complete the following:"  
    print "Enter the type of data used in the evaluation:"
    TYPE = raw_input(" >>  ")
    
    print "The base INPUT directory is:", INPUT+TYPE
    print "and OUTPUT directory is:", OUTPUT+TYPE
    print "Would you like to change it? y or n" 
    ch = raw_input(" >>  ").lower()
    if ch == 'y':
        print "Enter the base path for the INPUT directory (without the type):"
        INPUT = raw_input(" >>  ")
        if not os.path.isdir(INPUT):
            print "Error: Please try again; INPUT directory:"
            INPUT = raw_input(" >>  ")
            
        print "And enter the base path for evaluation OUTPUT directory (without the type):"
        OUTPUT = raw_input(" >>  ")
        if not os.path.isdir(OUTPUT):
            print "Error: Please try again; evaluation OUTPUT directory:"
            OUTPUT = raw_input(" >>  ")
        
#########
    print "Please choose a method to evaluate:"
    print "1. Holt-Winters"
    print "2. Auto-regression"
    print "3. 1st Markov chain"
    print "4. 2nd Markov chain"
    print "5. PRESS"
    print "6. Agile"
    print "7. FFNN Model"
    print "8. RNN Model"
    print "9. Entwine Model"
    print "10. Moving Average"
    print "11. Average combo4 Model"
    print "12. FFNN combo4 Model"
    print "13. Weighted Average model"
    print "0. Quit"
    choice = raw_input(" >>  ")
    ch = choice.lower();
    
    if ch == '':
        menu_actions['main_menu']()
    elif ch == '0':
        exit()
    else:
        METHOD = methods_dict[ch]

        pool = ThreadPool(4)
        files =  fileutils.getFilelist(INPUT+TYPE)
                
        params = []  
        if METHOD =='fnn':
            if TYPE.startswith("memory"):
                hyperpath = "../data/"+TYPE.replace("memory", "cpu")+"_networks/hyperparams.csv"
            else:
                hyperpath = "../data/"+TYPE+"_networks/hyperparams.csv"
            hyperparms =  np.genfromtxt(hyperpath, delimiter=',', dtype=None)
            for curRow in hyperparms:
                if TYPE.startswith("memory"):
                    params.append([INPUT+TYPE+'/'+curRow[0].replace("cpu", "memory").strip("'")+".csv", METHOD, TYPE, OUTPUT, INPUT, curRow[3], curRow[4]])
                else:
                    params.append([INPUT+TYPE+'/'+curRow[0].strip("'")+".csv", METHOD, TYPE, OUTPUT, INPUT, curRow[3], curRow[4]])
                    
        elif METHOD =='rnn':
            
            if TYPE.startswith("memory"):
                hyperpath = "../data/"+TYPE.replace("memory", "cpu")+"_rnn_networks/hyperparams.csv"
            else:
                hyperpath = "../data/"+TYPE+"_rnn_networks/hyperparams.csv"
            hyperparms =  np.genfromtxt(hyperpath, delimiter=',', dtype=None)
            for curRow in hyperparms:
                if TYPE.startswith("memory"):
                    params.append([INPUT+TYPE+'/'+curRow[0].replace("cpu", "memory").strip("'")+".csv", METHOD, TYPE, OUTPUT, INPUT, curRow[3], curRow[4]])
                else:
                    params.append([INPUT+TYPE+'/'+curRow[0].strip("'")+".csv", METHOD, TYPE, OUTPUT, INPUT, curRow[3], curRow[4]])
                    
        elif METHOD == 'entwine':
            hyperpath = "../data/entwine_networks/hyperparams.csv"
            hyperparms =  np.genfromtxt(hyperpath, delimiter=',', dtype=None)
            for curRow in hyperparms:
                params.append([INPUT+TYPE+'/'+curRow[0].strip("'")+".csv", METHOD, TYPE, OUTPUT, INPUT, curRow[3], curRow[4]])
                
        else:
            for f in files:
                params.append([f, METHOD, TYPE, OUTPUT, INPUT])
        
        if METHOD == 'avg4' or METHOD == 'combo4' or METHOD == 'wa':
#             ensembleModel(params[0])
            pool.map(ensembleModel,params)
            pool.close()
            pool.join()
        else:
#             print "skip"
            performsSlidingWindowForecast(params[0])
#             pool.map(performsSlidingWindowForecast, params)
#             pool.close()
#             pool.join()
#                    
#         pool = ThreadPool(4)
#         results = pool.map(performEvaluations, params)
#         pool.close()
#         pool.join()
#         fileutils.writeCSV(OUTPUT+"results/"+TYPE+"_"+METHOD+".csv", results)
#         print METHOD+" "+ TYPE + " complete"
          
        exit()
# Main Program
if __name__ == "__main__":
    
    main()