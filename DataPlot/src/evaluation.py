'''
Created on 07 Nov 2014

@author: Manrich
'''
from __future__ import division
import tsutils
import numpy as np
from sklearn.metrics import mean_squared_error

def calc_RMSE(y_true, y_fc, print_=False):
    '''
    Calculates the Root Mean Squared Error (RMSE) per sample in y_true compared to y_fc
    
    Params
    ------
    @param y_true: The true values
    @type y_true: array-like
    @param y_fc: Forecasted values
    @type y_fc: array-like
    @param print_: Print confusion matrix and Accuracy rates
    @type print_: boolean
    
    Return
    ------
    @return: The RMSE 
    @rtype: float 
    '''
    rmse = np.sqrt(mean_squared_error(y_true, y_fc))
    
    if print_:
        print "RMSE = ", rmse
        
    return rmse
    
def calc_upper_lower_acc(y_true, y_fc, band=0.1, print_=False):
    '''
    Calculates the 10% under and over estimation count
    
    Params
    ------
    @param y_true: The true values
    @type y_true: array-like
    @param y_fc: Forecasted values
    @type y_fc: array-like
    @param band: Percentage band use to calculate the upper and lower bounds
    @type band: float 
    @param print_: Print confusion matrix and Accuracy rates
    @type print_: boolean
    
    Return
    -------
    @return: correct prediction rate, over-estimation rate, under-estimation rate]
    @rtype: float
    '''
    
    upper10 = (1+band) * y_true
    lower10 = (1-band) * y_true

    correct = sum(np.logical_and(np.less_equal(y_fc,upper10),np.greater_equal(y_fc,lower10)))
    overest = sum(np.greater_equal(y_fc,upper10))
    underest = sum(np.less_equal(y_fc,lower10))
    
    correct_rate = np.divide(1.0*correct, correct+overest+underest)
    overest_rate = np.divide(1.0*overest, correct+overest+underest)
    underest_rate = np.divide(1.0*underest, correct+overest+underest)
    
    if print_:
        print "Correct prediction rate =", correct_rate
        print "Over-estimation rate =", overest_rate
        print "Under-estimation rate =", underest_rate

    return correct_rate, overest_rate, underest_rate

def calc_persample_accuracy(y_true, y_fc, threshold, print_=False):
    '''
    Calculates the True positive rate, False postive rate per sample accuracy
    (Using True postive, True negative, False positive, False negative [Tp, Tn, Fp, Fn] 
    respectively)
    
    Params:
    ------
    @param y_true: The true values
    @type y_true: array-like
    @param y_fc: Forecasted values
    @type y_fc: array-like
    @param threshold: A sample is classified as overload when above threshold
    @type threshold: float
    @param print_: Print confusion matrix and Accuracy rates
    @type print_: boolean
    
    Return
    -------
    @return: True positive rate, False positive rate
    @rtype: float
    '''
    r_overload_samples = y_true>threshold
    fc_overload_samples = y_fc>threshold
    
    Tp = sum(np.logical_and(r_overload_samples, fc_overload_samples))
    Tn = sum(np.logical_and(~r_overload_samples, ~fc_overload_samples))
    Fp = sum(np.logical_and(~r_overload_samples, fc_overload_samples))
    Fn = sum(np.logical_and(r_overload_samples, ~fc_overload_samples))
    
    # Calculate rates
    TPR = np.divide(1.0*Tp, Tp+Fn)
    FPR = np.divide(1.0*Fp, Fp+Tn)
    
    if print_:
        printConfusion(Tp, Fp, Tn, Fn)
    return TPR, FPR

def calc_overload_states_acc(y_true, y_fc, threshold, overload_dur=5, print_=False):
    '''
    Calculates the Accuracy rates for the overload and not-overload states, where an overload
    state is defined as samples being above the threshold for at least overload_dur samples
    
    Params:
    -------
    @param y_true: The true values
    @type y_true: array-like
    @param y_fc: Forecasted values
    @type y_fc: array-like
    @param threshold: A sample is classified as overload when above threshold
    @type threshold: float
    @param print_: Print confusion matrix and Accuracy rates
    @type print_: boolean
    
    Return
    -------
    @return: True positive rate, False positive rate
    @rtype: float
    '''
    r_overload_states = overload_states_where(y_true>threshold, overload_dur)
    fc_overload_states = overload_states_where(y_fc>threshold, overload_dur)

    Tp, total_p = count_edges_within_band(r_overload_states, fc_overload_states)
    Tn, total_n = count_edges_within_band(r_overload_states, fc_overload_states, rising=False)
    Fp = total_p-Tp
    Fn = total_n-Tn
    
    # Calculate rates
    TPR = np.divide(1.0*Tp, Tp+Fn)
    FPR = np.divide(1.0*Fp, Fp+Tn)

    if print_:
        printConfusion(Tp, Fp, Tn, Fn)

    return TPR, FPR


def overload_states_where(condition, dur):
    '''
    Determines where the condition is True for at least dur samples
    
    Params
    ------
    @param condition: array containing conditions where true
    @type condition: Boolean array
    @param dur: The least number of samples where the condition should be True
    @type: int
    
    Returns:
    -------
    @returns: A reconstructed boolean array containing only valid states length of condition
    @rtype: Boolean array
    '''
    idx = []
    state_matrix = []
    
    # Find valid overload state indexes
    for start, stop in tsutils.contiguous_regions(condition):
        if stop - start >= dur:
            idx.append([start,stop])
    idx = np.array(idx)
    
    # Build state matrix
    prev_index = 0
    for i in range(idx.shape[0]):
        state_matrix.append([0]*(idx[i,0] - prev_index))
        state_matrix.append([1]*(idx[i,1] - idx[i,0]))
        prev_index = idx[i,1]
    state_matrix.append([0]*(len(condition) - prev_index))
    
    return np.array(sum(state_matrix,[]))

def count_edges_within_band(a, b, band=3, rising=True):
    '''
    Counts the number of rising (or falling) edges match, within a sample band
    
    Params
    -------
    @param a, b: Arrays that will be compared
    @type a, b: Boolean array 
    @param band: The number of samples of tolerance
    @type band: float
    @param rising: Specify rising or falling edge
    @type rising: boolean 
    
    Returns
    -------
    @return: Count of matching edges, total true rising (or falling) edges
    @rtype: int
    '''
    if rising:
        a = np.r_[a[0], np.diff(a)]>0
        b = np.r_[b[0], np.diff(b)]>0
    else:
        a = np.r_[a[0], np.diff(a)]<0
        b = np.r_[b[0], np.diff(b)]<0

    total_edges = sum(a)
    result = np.logical_and(a, b)
    for offset in np.add(range(3),1):
        posoff = np.r_[[0]*offset, np.logical_and(a[:-offset], b[offset:])]
        negoff = np.r_[np.logical_and(a[offset:], b[:-offset]), [0]*offset]
        result = np.logical_or(result, posoff)
        result = np.logical_or(result, negoff)

    return sum(result), total_edges


def printConfusion(Tp, Fp, Tn, Fn):
    '''
    Prints the confusion matrix using the True positive, False positive, True negative
    and False negative values. Also calculates and prints the True Positive Rate (TPR) and
    False Positive Rate (FPR)
    '''
    print " ------------"
    print "|", Tp, "|", Fp, "|"
    print "-----------"
    print "|", Fn, "|", Tn, "|"
    print " ------------"
    
    # Calculate rates
    TPR = np.divide(1.0*Tp, Tp+Fn)
    FPR = np.divide(1.0*Fp, Fp+Tn)

    print "TPR = ", TPR
    print "FPR = ", FPR