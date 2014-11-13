'''
Created on 03 Nov 2014

@author: Manrich
'''
import statsmodels.api as sm
import numpy as np
from scipy import signal

def findDominentSeason(y):
    '''
    Finds the dominant season in samples
    y: data
    '''
    N = len(y)
    
    freq, pwr = signal.periodogram(y)
    ind = np.argmax(pwr)
    
    return np.min([np.int_(np.round(1/freq[ind])), N])

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx