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


# Main definition - constants
menu_actions  = {}  
methods_dict = {}
INPUT = 'd:/data/'
OUTPUT = 'd:/data/'
TYPE = None
METHOD = None


def performsSlidingWindowForecast(params, minpercentile=5, step=30, input_window=3000, predic_window=30):
    '''
    Input window = 250 hours = 250*12 = 3000 
    look ahead window 60 samples =  5 hours = 720min/5 = 60
    
    '''
    filename, METHOD, TYPE, OUTPUT = params[0:4]
#Wikidata
#     data = np.genfromtxt(filename)
#     data = data/np.max(data)

    data = np.nan_to_num(np.genfromtxt(filename, delimiter=',', skip_header=1)[:,1]).ravel()
    minimum = np.percentile(data,minpercentile)
    N = len(data)
    result = []
    print filename, "started..."
    for strIndex in range(0,N-input_window - predic_window, step):
        if strIndex == 0:
            y = data[:input_window]
            if METHOD == 'ar':
                model = AR_model.AR_model(y, order=30)
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
                model = Wavelet_model(y)
            elif METHOD == 'fnn':
                filename, METHOD, TYPE, OUTPUT, INPUT, curEta, curLmda = params[:7]
                curMachine = filename.split('/')[-1]
                if TYPE.startswith("memory"):
                    curMachine = curMachine.replace("memory", "cpu")
                    model = Fnn_model.Fnn_model(data=data, machineID = curMachine, netPath="../data/"+TYPE.replace("memory", "cpu")+"_networks/"+curMachine.replace(".csv",".xml"), eta=curEta, lmda=curLmda)
                else:
                    model = Fnn_model.Fnn_model(data=data, machineID = curMachine, netPath="../data/"+TYPE+"_networks/"+curMachine.replace(".csv",".xml"), eta=curEta, lmda=curLmda)
            elif METHOD == 'rnn':
                filename, METHOD, TYPE, OUTPUT, INPUT, curEta, curLmda = params[:7]
                curMachine = filename.split('/')[-1]
                if TYPE.startswith("memory"):
                    curMachine = curMachine.replace("memory", "cpu")
                    model = Fnn_model.Fnn_model(data=data, machineID = curMachine, netPath="../data/"+TYPE.replace("memory", "cpu")+"_networks/"+curMachine.replace(".csv",".xml"), eta=curEta, lmda=curLmda)
                else:
                    model = Fnn_model.Fnn_model(data=data, machineID = curMachine, netPath="../data/"+TYPE+"_rnn_networks/"+curMachine.replace(".csv",".xml"), eta=curEta, lmda=curLmda)
            elif METHOD == 'entwine':
                filename, METHOD, TYPE, OUTPUT, INPUT, curEta, curLmda = params[:7]
                curMachine = filename.split('/')[-1]
                data2 = np.nan_to_num(np.genfromtxt(filename.replace("cpu", "memory"), delimiter=',', skip_header=1)[:,1]).ravel()
                
                model = Entwine_model.Entwine_model([data, data2], machineID = curMachine, netPath="../data/entwine_networks/"+curMachine.replace(".csv",".xml"), eta=curEta, lmda=curLmda)
                
            model.fit()
        else:
            if METHOD == 'press' or METHOD == 'markov':
                y = data[input_window + strIndex - step:input_window + strIndex]
            else:
                y = data[strIndex:strIndex+input_window]
            model.update(y)
              
        y_pred = np.atleast_2d(model.predict(predic_window))
        y_pred[y_pred[:,0]<0,0] = minimum
        result.append(y_pred[:,0])
    f = filename.split('/')[-1]
    fileutils.writeCSV(OUTPUT+TYPE+"_"+METHOD+"/"+f, np.atleast_2d(result))
    print filename, "complete!"
    
def performEvaluations(params, train_window = 3000, overload_dur = 5, overload_percentile = 70, steps=30):
    
    filename, METHOD, TYPE, OUTPUT, INPUT = params[:5]
    filename = filename.split('/')[-1]
    
    cur_results = []
    forecasts = np.nan_to_num(np.genfromtxt(INPUT+TYPE+"_"+METHOD+"/" + filename, delimiter=',',usecols=range(0,30))).ravel() # ,usecols=range(0,30)
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
    '2': 'ar',
    '3': 'markov1',
    '4': 'markov2',
    '5': 'press',
    '6': 'agile',
    '7': 'fnn',  
    '8': 'rnn', 
    '9': 'entwine', 
    '10': 'ma',        
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
            
#         performsSlidingWindowForecast(params[0])
        pool.map(performsSlidingWindowForecast, params)
        pool.close()
        pool.join()
            
        pool = ThreadPool(4)
        results = pool.map(performEvaluations, params)
        pool.close()
        pool.join()
        fileutils.writeCSV(OUTPUT+"results/"+TYPE+"_"+METHOD+".csv", results)
        print METHOD+" "+ TYPE + " complete"
          
        exit()
# Main Program
if __name__ == "__main__":
    
    main()