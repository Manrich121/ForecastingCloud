'''
Created on 03 Nov 2014

@author: Manrich
'''
from scipy import signal

import numpy as np
import scipy.stats
import matplotlib.patches as mpatches

def findDominentSeason(y):
    '''
    Finds the dominant season in samples
    y: data
    '''
    N = len(y)
    
    freq, pwr = signal.periodogram(y)
    ind = np.argmax(pwr[1:])+1
    
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

def elp(pltdic, best='min',skiplast=0):
    miny = np.inf
    mindx = 0
    maxy = -np.inf
    maxdx = 0
    for i in range(len(pltdic['means'])-skiplast):
        val = pltdic['means'][i].get_ydata()[0]
        if  val < miny:
            miny = val
            mindx = i
        if val > maxy:
            maxy = val
            maxdx = i

    if best == 'min':
        y1,y2 = pltdic['caps'][mindx*2].get_ydata()[0], pltdic['caps'][mindx*2+1].get_ydata()[0]
        ellipse1 = mpatches.Ellipse((mindx+1.0, (y1+y2)/2), 0.7, (y2-y1)*1.5, fill=False, ec='green',linewidth=1.5)

        y1,y2 = pltdic['caps'][maxdx*2].get_ydata()[0], pltdic['caps'][maxdx*2+1].get_ydata()[0]
        ellipse2 = mpatches.Ellipse((maxdx+1.0, (y1+y2)/2), 0.7, (y2-y1)*1.5, fill=False, ec='red',linewidth=1.5, linestyle='dashed')
    else:
        y1,y2 = pltdic['caps'][maxdx*2].get_ydata()[0], pltdic['caps'][maxdx*2+1].get_ydata()[0]
        ellipse1 = mpatches.Ellipse((maxdx+1.0, (y1+y2)/2), 0.7, (y2-y1)*1.5, fill=False, ec='green',linewidth=1.5)

        y1,y2 = pltdic['caps'][mindx*2].get_ydata()[0], pltdic['caps'][mindx*2+1].get_ydata()[0]
        ellipse2 = mpatches.Ellipse((mindx+1.0, (y1+y2)/2), 0.7, (y2-y1)*1.5, fill=False, ec='red',linewidth=1.5, linestyle='dashed')

    return ellipse1, ellipse2

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

"""
Function definion: Takes the mean1 and stadard_deviation1, mean2 and stadard_deviation2 in arguments and returns the T-value
"""
def calcT(x1, s1, x2, s2, n):
    sp = np.sqrt(((n-1)*s1*s1 + (n-1)*s2*s2)/(2.0*n-2))
    return (x1 - x2)/(sp*np.sqrt(2.0/n))

def calcPvalue(data1, data2, N=500):
    x1, s1 = np.mean(data1), np.std(data1)
    x2, s2 = np.mean(data2), np.std(data2)
    print 'pval=', scipy.stats.t.sf(np.abs(calcT(x1, s1, x2, s2, N)), N-1)