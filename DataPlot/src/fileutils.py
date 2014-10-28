'''
Created on 12 Sep 2014

@author: Manrich
'''
import csv
import gzip
import os

import numpy as np


def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=np.float):
    def iter_func():
        if filename.endswith('gz'):
            infile = gzip.open(filename, 'r')
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


def getCsvRows(filename):
    if filename.endswith('gz'):
        csvfile = gzip.open(filename, 'rb')
    else:
        csvfile = open(filename, "rb")
    datareader = csv.reader(csvfile)
    for row in datareader:
        yield row

def getFilelist(filepath):
    files_out = []
    
    for root, _ ,files in os.walk(filepath):
        for file in files:
            files_out.append(root+'/'+ file)
            
    return files_out

def writeCSV(filename, data=[], mode='wb', header=None):
    with open(filename, mode) as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        if header != None:
            csvwriter.writerow(header)
        for row in data:
            csvwriter.writerow(row)
        
    