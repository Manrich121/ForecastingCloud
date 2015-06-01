'''
Created on 12 Sep 2014

@author: Manrich
'''

from matplotlib import pyplot as plt

from StringIO import StringIO
import fileutils
import numpy as np
from test.test_os import resource
import scipy

machines1 = ['4155527081','329150663','3938719206','351618647','431052910','257348783',
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

machines2 = ['1093692','1094814','1189733520','124596184','1268205','1269993','1271669',
        '1272048','1272600','1273710','1275580','1301871','1331993','1335648',
        '1338945','1340388','1429190047','1436297879','1436506108','1436808073',
        '1438242528','1439829698','16915959','1697583','1726343263','1726343696',
        '182859408','2055507446','2055737168','2107213808','2198001960',
        '2274895763','2511817074','257335661','257348783','257499345','257500341',
        '257512310','2595183881','277449078','2851850','288787745','2899249543',
        '294771499','294816718','294847197','294901680','294908001','294949854',
        '3002780374','30787958','30790115','317471625','317483691','317488701',
        '317497354','317499484','3215724643','3250893612','329150663','3294802334',
        '3325378525','3337876230','3349962080','336021543','336025676','336037285',
        '336055347','336074128','337885583','3405236527','351618647','351621284',
        '351635981','353713140','3539540276','3550322224','3652042894','368695780',
        '370130410','376217870','376271364','3813488043','38643517','38653355',
        '3894543095','39146853','3938719206','3938826162','4081219342',
        '4155527081','4246468145','4302612937','4304175952','4304743890',
        '431081448','4346222838','4469943288','4469994137','4476762131']

machines3 = ['1017271853', '1093202', '1093700', '1094094', '1094240', '1094553', 
             '1094605', '1095307', '115400882', '1268820', '1269577', '1270422', 
             '1270903', '1271016', '1272056', '1302353', '1331690', '1331713', '1333882', 
             '1335764', '1339089', '1339145', '1340562', '1436331297', '1436457296', 
             '1436514310', '1436793141', '1464042314', '1602467098', '1666591', '16916378', 
             '2006362021', '2197275471', '2248361096', '227433054', '2347183346', 
             '2381577379', '2421729495', '2568509112', '257333194', '257335594', 
             '257338196', '257338292', '257388875', '257408764', '2781752128', '2849392', 
             '288908994', '288943705', '294772488', '294933859', '295014033', '3001867270', 
             '3013143852', '30788565', '30790107', '317468385', '317487213', '317497174', 
             '317497833', '3241168570', '336037217', '336055587', '336071716', '340153818', 
             '351619555', '351651939', '3652042894', '3813489575', '38643416', '38661859', 
             '38663831', '38683944', '38692013', '3890224606', '3938677503', '3938913241', 
             '3949007738', '400426685', '4203755923', '431035532', '4469151537', '4477559343', 
             '462075712', '4802145976', '4802495635', '4820093820', '4820157039', 
             '4820227126', '5225625303', '563849022', '63683748', '6608772', '662382', 
             '711962', '765926', '903225', '904306', '907255', '97966846']

machines4 = ['10486091', '1092958', '1093423', '1093559', '1093972', '1095263', '124596213', '124625221', '125627111', '1268752', '1268904', '1270655', '1271015', '1271272', '1271936', '1272058', '1272597', '1273030', '1273130', '1273854', '1275154', '12933671', '13010154', '1301828', '1324363', '1324478', '1331892', '1333840', '1335142', '1339142', '1404437', '1419555512', '1423369981', '1429206925', '1436299544', '1436304298', '1436331334', '1436360167', '1436368564', '1436474411', '1436486817', '1436487910', '1436489990', '1436697174', '1438221455', '1438817626', '1439104436', '1439177114', '1511782180', '16917838', '16918683', '1697626', '17004853', '17216304', '1912218991', '2054984604', '2055342510', '207999948', '2096233187', '2096989215', '2097465275', '2110935011', '216969487', '217519477', '227394150', '227417991', '227434848', '228617662', '2347917475', '2357190483', '2420922114', '2424313575', '2508447313', '2568167487', '257337110', '257340267', '257349129', '257351351', '257406818', '257407416', '257409979', '257410942', '257500738', '257502329', '257503193', '2601422167', '2634467021', '2643511514', '277432440', '277433540', '284799741', '2852069', '285290147', '288787745', '288806457', '288953708', '294771500', '294793958', '294823356', '294847235', '294862576', '294862969', '294879925', '294994569', '294995400', '2990346970', '30788567', '30789498', '308518466', '3143806165', '317330972', '317468439', '317469422', '317469463', '317477827', '317486724', '317488580', '317489271', '317489295', '317490164', '317495556', '317495588', '317504398', '317504846', '317808274', '32064172', '32076357', '3231024074', '323143785', '3251992992', '328878058', '328892793', '3305896582', '332582575', '3338239375', '3351129364', '3352022725', '3352280010', '336021526', '336025725', '336026201', '336030258', '336030356', '336036419', '336038559', '336038560', '336041201', '336047452', '336047701', '336051750', '336059045', '336068551', '336074127', '3405227965', '3407328369', '347828743', '350588109', '351594694', '351604308', '351612196', '351613868', '351617798', '351619550', '351625737', '351627428', '351640926', '351642433', '351652245', '351662258', '353712197', '3571700224', '3584822123', '359920779', '3605330708', '3622440393', '3641115797', '3675871662', '3676306487', '367886065', '3739340729', '376216899', '376288168', '376328657', '379050397', '381236', '3813002360', '3821140990', '3829598', '3858903110', '38644180', '38657132', '38673130', '38680139', '38683247', '38692360', '38698665', '38708171', '38742685', '3890625724', '400426675', '41', '410695', '4205181815', '421593', '4246003971', '4286186493', '4302215883', '4302787106', '4304096877', '4304154151', '4304169803', '4304730899', '431035538', '431038863', '4469366776', '4469970863', '4476549437', '4477658597', '46295811', '478267555', '4802475937', '4802500501', '4802905544', '4820029939', '4820061208', '4820076231', '4820091923', '4820094953', '4820110705', '4820131656', '4820138112', '4820139430', '4820236249', '5015682200', '5068065359', '537747028', '554297904', '554338230', '564504416', '567345', '5743437955', '5781238', '5783206', '587055228', '597205677', '599619', '601132397', '61730926', '63625913', '640687632', '643114551', '6565029', '6565187', '6566266', '6567448', '6567630', '6567815', '6568432', '6607759', '6607771', '6608677', '6640621', '6641575', '6641712', '672224', '672303', '681910', '682785', '705747', '705775', '708463900', '708480354', '708711467', '710941', '711981', '717263', '765976', '7777953', '7781386', '8055696', '8055706', '82755269', '854812624', '906109', '907206', '907396', '908056', '908315', '908319', '924386824', '924486088', '930024438', '942456165', '97922776', '97923398', '988440', '988699', '988786']

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
                fileutils.writeCSV("d:/data/perMachine5/"+machine+".csv", machineUsage[machine])
            else:
                fileutils.writeCSV("d:/data/perMachine5/"+machine+".csv", machineUsage[machine], mode='ab')
                 
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
  
if __name__ == '__main__':
#     ids = np.genfromtxt('d:/data/machineIDs.csv',dtype=str) 
#     machines = machines4[100:200]
#     extractMachineData()

    for f in fileutils.getFilelist("D:/data/perMachine5"):
        print f
        readAndAggregate(f, "d:/data/cpu5", 'cpu')
        readAndAggregate(f, "d:/data/memory5", 'memory')
