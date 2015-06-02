'''
Created on 01 Jun 2015

@author: Manrich
'''

import fileutils
import numpy as np
import sys, os
import StringIO
import AR_model
import HW_model
import Press_model
import Wavelet_model
import Markov_model

from multiprocessing import Pool as ThreadPool 
 
# Main definition - constants
menu_actions  = {}  
methods_dict = {}
INPUT = 'd:/data/'
OUTPUT = 'd:/data/'
TYPE = None
METHOD = None


def performsSlidingWindowForecast(params, minpercentile=5, step=30, input_window=3000, predic_window=30, order_=1):
    '''
    Input window = 250 hours = 250*12 = 3000 
    look ahead window 60 samples =  5 hours = 720min/5 = 60
    
    '''
    filename, METHOD, TYPE, OUTPUT = params
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
            elif METHOD == 'hw':
                model = HW_model.HW_model(y, minimum, 'additive')
            elif METHOD == 'markov':
                model = Markov_model.Markov_model(y, maximum=max, order=order_)
            elif METHOD == 'press':
                model = Press_model.Press_model(y, maximum=max)
            elif METHOD == 'agile':
                model = Wavelet_model(y)
            model.fit()
        else:
            if METHOD == 'press' or METHOD == 'markov':
                y = data[input_window + strIndex - step:input_window + strIndex]
            else:
                y = data[strIndex:strIndex+input_window]
            model.update(y)
              
        y_pred = model.predict(predic_window)
        y_pred[y_pred[:,0]<0,0] = minimum
        result.append(y_pred[:,0])
    f = filename.split('/')[-1]
    fileutils.writeCSV(OUTPUT+TYPE+"_"+METHOD+"/"+f, np.atleast_2d(result))
    print filename, "complete!"
    
# Back to main menu
 
# Exit program
def exit():
    print "Goodbye :)"
    sys.exit()
 
methods_dict = {
    '1': 'hw',
    '2': 'ar',
    '3': 'markov',
    '4': 'press',
    '5': 'agile',            
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
    print "3. Markov chain"
    print "4. PRESS"
    print "5. Agile"
#     print "6. Combo: Average Model"
#     print "7. Combo: FFNN Model"
#     print "0. Back"
    print "9. Quit"
    choice = raw_input(" >>  ")
    ch = choice.lower();
    
    if ch == '':
        menu_actions['main_menu']()
    elif ch == '9':
        exit()
    else:
        METHOD = methods_dict[ch]

        pool = ThreadPool(4)
        files =  fileutils.getFilelist(INPUT+TYPE)
        
        params = []
        
        for f in files:
            params.append([f, METHOD, TYPE, OUTPUT])
        
        
#         performsSlidingWindowForecast(files[0])
        pool.map(performsSlidingWindowForecast, params)
        pool.close()
        pool.join()
    
 
# Main Program
if __name__ == "__main__":
    main()