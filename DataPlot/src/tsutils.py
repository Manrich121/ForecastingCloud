'''
Created on 03 Nov 2014

@author: Manrich
'''
import statsmodels.api as sm
import numpy as np

def findDominentSeason(y, ignoreDC=True):
    '''
    Finds the dominant season in samples
    y: data
    ignoreDC: set to include DC component or not
    '''
    N = len(y)
    yf = sm.tsa.stattools.periodogram(y)
    xf = np.linspace(0,1,N/2) 
    
    strIndex = 0
    if ignoreDC:
        strIndex = 1
    
    ibest = np.argmax(yf[strIndex:N/2]) + strIndex
    
    return np.min([np.int_(np.round( 1.0/xf[ibest])), N])

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