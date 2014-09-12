'''
Created on 12 Sep 2014

@author: Manrich
'''
import numpy as np
import gzip

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=np.float):
    def iter_func():
        if filename.endswith('gz'):
            infile = gzipopen(filename, 'r')
        else:
            infile = open(filename, 'r')
        for _ in range(skiprows):
            next(infile)
        for line in infile:
            line = line.rstrip().split(delimiter)
            for item in line:
                if item != '':
                    yield dtype(item)
                else:
                    item = 0;
                    yield
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data