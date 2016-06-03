

import OlinGraph

import qrtools


def readNodes(mapFile):
    """Takes in a filename for a occupancy-grid map graph, and it reads
    in the data from the file. It then generates the map appropriately."""
    try:
        filObj = open(mapFile, 'r')
    except:
        print("ERROR READING FILE, ABORTING")
        return
    sawFirstLine = False
    readingNodes = False
    allData = []
    numNodes = -1
    row = -1
    for line in filObj:
        line = line.strip()
        if line == "" or line[0] == '#':
            continue
        elif not sawFirstLine:
            numNodes = int(line.strip())
            sawFirstLine = True
            readingNodes = True
            row = 0
        elif readingNodes and row < numNodes:
            (num, pos, descr) = line.split("   ")
            allData.append( (descr, line))
            row += 1
            if row == numNodes:
                break
    return allData



            
nodes = readNodes("olinGraph.txt")
print nodes
for node in nodes:
    (descr, dataStr) = node
    fName = descr.replace(" ", "")
    code = qrtools.QR(data = dataStr, data_type="text")
    print code.data_type
    print code.data
    res = code.encode(filename= fName) 
    print "GOT HERE"
    print res
    print code.filename
    