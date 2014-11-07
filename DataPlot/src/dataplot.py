'''
Created on 12 Sep 2014

@author: Manrich
'''

from matplotlib import pyplot as plt

from StringIO import StringIO
import fileutils
import numpy as np
from test.test_os import resource

machines = ['4155527081','329150663','3938719206','351618647','431052910','257348783',
     '5655258253','3550322224','1303745','3894543095','336025676','3405236527',
     '431081448','84899647','1268205','778602858','351621284','317488701',
     '2595183881','621588868','4304743890','3938826162','351635981',
     '5656207113','2055737168','317499484','2055507446','1335648','662205',
     '1436348411','717319','1301871','124596184','2274895763','4469213375',
     '336055347','5796442','6567274','294771706','336030391',
     '336045945','6955600','866746667','1442486585','376211539','5782232',
     '5781488','6570572','227414872','6608763','3233542229','672515',
     '336021545','564444639','6640648','1338948','1437072475','294973335',
     '1331706','336036882','294887209','1335782','6567284','410058079',
     '155313295','4802475358','431038861','907812','82732361','2912464652',
     '4820238819','905814','1439356567','84847796','329144835','1269048',
     '348769060','904514','318418982','38655581','344595418','4217136868',
     '4469371300','1095481','167553463','7753127','376751173','16915982',
     '1664088958','905062','2549393774','660404','3349189108','711355',
     '257395954','323143631','6640087','1695367','336045944','3858945898', '4815459946']

def extractMachineData():
    datafiles = fileutils.getFilelist("D:/googleClusterData/clusterdata-2011-1/task_usage")
    machineUsage = {}
    
    startAt = 0;
    
    for machine in machines:
        machineUsage[machine] = []
    
    for datafile in datafiles[startAt:]:
        print datafile
        for row in fileutils.getCsvRows(datafile):
            curMachine = row[4]
            if curMachine in machines:
                machineUsage[curMachine].append(row)
    
        for machine in machineUsage.keys():
            if startAt == 0:
                fileutils.writeCSV("d:/data/"+machine+".csv", machineUsage[machine])
            else:
                fileutils.writeCSV("d:/data/"+machine+".csv", machineUsage[machine], mode='ab')
                 
        startAt += 1         
        machineUsage.clear()
        for machine in machines:
            machineUsage[machine] = []
'''
    Extracts a resource type from a google cluster data file
'''
def readAndAggregate(filename, outputDir, resource='cpu'):
    colomn = 1 # start time
    if resource == 'cpu':
        colomn = 5;
    elif resource == 'memory':
        colomn = 7
    elif resource == 'diskIO':
        colomn = 11 
    
    resourcePerTask = np.genfromtxt(filename, delimiter=',', skiprows=0, usecols=(0,colomn), filling_values = '0')
    
    fileCsv = filename.split('/')[-1]
    
    strTime = 600e6
    endTime = strTime + 300e6
    globalEndTime = 2506200000000
    
    numberOfRows = globalEndTime/300e6
    aggregatedData = np.zeros([numberOfRows, 2], dtype=float)
    x = 0
    
    aggregatedData[x,0] = strTime
    
    for row in resourcePerTask[:]:
        time = np.float_(row[0])
        if (time>=strTime and time<endTime):
            row_f = np.float_(row[1])
            aggregatedData[x,1] += row_f
            
        else:
            strTime = endTime
            endTime += 300e6
            x +=1
            aggregatedData[x,0] = strTime
            
    while x < numberOfRows:
        strTime = endTime
        endTime += 300e6
        aggregatedData[x,0] = strTime
        x +=1
        
    fileutils.writeCSV(outputDir+'/' + resource + '_' +fileCsv, aggregatedData, header=('Time',resource.capitalize()))

def readAndWriteAllCpuRate():
    for f in fileutils.getFilelist("D:/data/perMachine"):
        print f
        readAndAggregate(f, "d:/data/cpuRate", 'cpu')
    
if __name__ == '__main__':
    for f in fileutils.getFilelist("D:/data/perMachine"):
        print f
        readAndAggregate(f, "d:/data/diskIO", 'diskIO')
    
         
    data = np.genfromtxt("d:/data/cpuRate/cpu_907812.csv", delimiter=',', usecols=(0,1))
    plt.plot(data[:,0]/1e6,data[:,1])
    plt.title("Machine 907812's cpu over 29 day period")
    plt.xlabel("Time (s)")
    plt.ylabel("Cpu time")
    plt.show()