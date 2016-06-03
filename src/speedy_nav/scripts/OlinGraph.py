
######################################
# Graph of East part of Olin-Rice, 2nd floor
#
"""Contains a graph of meaningful waypoints"""

import math

# For BFS/DFS, use the following import line
#from Graphs import ListGraph
# For Best-First or A*, use the following import line
from MapGraph import *


def readMapFile(mapFile):
    """Takes in a filename for a occupancy-grid map graph, and it reads
    in the data from the file. It then generates the map appropriately."""
    try:
        filObj = open(mapFile, 'r')
    except:
        print("ERROR READING FILE, ABORTING")
        return
    sawFirstLine = False
    readingNodes = False
    readingEdges = False
    allData = []
    numNodes = -1
    row = -1
    graph = None
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
            [nodeNumStr, locStr, descr] = line.split("   ")
            nodeNum = int(nodeNumStr)
            if nodeNum != row:
                print "ROW DOESN'T MATCH"
            else:
                dataList = locStr.split()
                nodeData = [part.strip("(),") for part in dataList]
                allData.append( (float(nodeData[0]), float(nodeData[1])) )
            row += 1
            if row == numNodes:
                readingNodes = False
                readingEdges = True
                graph = MapGraph(numNodes, allData)
                
        elif readingEdges:
            [fromNode, toNode] = [int(x) for x in line.split()]
            graph.addEdge(fromNode, toNode)
        else:
            print "Shouldn't get here", line
    return graph
            
            
     
olin = readMapFile("olinGraph.txt")


## The node data has locations based on a grid overlay, where
## (0,0) is outside the building near the southeast entrance, and
## distances are in meters.  The x axis runs north-south, the y
## axis runs east-west
#nodeLocs = [(22.2, 4),  # Home lab   Node 0
        #(22.2, 6.5),  # Lab hall   Node 1
        #(20.0, 6.5),  # East T   Node 2
        #(31.2,6.5),  # East L   Node 3
        #(12.9, 6.5),  # 259 hall   Node 4
        #(8.7, 6.1),  # Right Angle door   Node 5
        #(20.0, 9.9),  # 258 lab hall   Node 6
        #(17.3, 10.2),  # 258 lab   Node 7
        #(20.0, 17.9),  # 256 lab hall   Node 8
        #(17.3, 18.2),  # 256 lab   Node 9
        #(20.0, 22.4),  # 250 center hall   Node 10
        #(20.0, 26.1),  # Reading room hall   Node 11
        #(17.3, 26.3),  # Reading room   Node 12
        #(20.0, 27.6),  # 205 center hall    Node 13
        #(20.0, 40.1),  # Atrium center hall   Node 14
        #(23.8, 21.2),  # 250 center   Node 15
        #(28.1, 21.2),  # 250 north   Node 16
        #(25.0, 29.5),  # 205 east   Node 17
        #(23.2, 38.1),  # 205 west   Node 18
        #(31.0, 10.6),  # 247 hall   Node 19
        #(34.2, 10.0),  # 247 room   Node 20
        #(31.0, 18.6),  # 245 hall   Node 21
        #(35.0, 18.0),  # 245 room   Node 22
        #(31.0, 22.6),  # 250 north hall   Node 23
        #(31.0, 26.7),  # 243 hall   Node 24
        #(35.0, 26.2),  # 243 room   Node 25
        #(31.0, 28.6),  # 205 north hall   Node 26
        #(31.0, 38.9),  # 241 hall   Node 27
        #(35.1, 38.6),  # 241 room   Node 28
        #(31.0, 40.1),  # Atrium north hall   Node 29
        #(6.7, 10.1),  # Right Angle center   Node 30
        #(3.2, 7.6),  # 233 room   Node 31
        #(6.1, 8.5),  # 232-233 hall   Node 32
        #(3.2, 9.4),  # 232 room   Node 33
        #(3.2, 13.6),  # 231 room   Node 34
        #(6.1, 14.5),  # 230-231 hall   Node 35
        #(3.2, 15.4),  # 230 room   Node 36
        #(3.2, 19.8),  # 229 room   Node 37
        #(6.1, 20.7),  # 228-229 hall   Node 38
        #(3.2, 21.6),  # 228 room   Node 39
        #(3.2, 25.9),  # 227 room   Node 40
        #(6.1, 26.8),  # 226-227 hall   Node 41
        #(3.2, 27.7),  # 226 room   Node 42
        #(3.2, 31.8),  # 225 room   Node 43
        #(6.1, 32.7),  # 224-225 hall   Node 44
        #(3.2, 33.6),  # 224 room   Node 45
        #(3.2, 38.1),  # 223 room   Node 46
        #(6.1, 39.0),  # 222-223 hall   Node 47
        #(3.2, 39.9),  # 222 room   Node 48
        #(6.3, 41.1),  # Atrium south hall   Node 49
        #(31.1, 44.8),  # Atrium NE   Node 50
        #(32.6, 50.6),  # North main doors   Node 51
        #(31.2, 57.1),  # Atrium NW   Node 52
        #(26.2, 50.7),  # Atrium NC   Node 53
        #(23.0, 43.2),  # 205 west atrium   Node 54
        #(20.1, 57.1),  # Atrium CW   Node 55
        #(19.6, 50.7),  # Atrium CC   Node 56
        #(20.0, 44.8),  # Atrium CE   Node 57
        #(14.1, 50.6),  # Atrium SC   Node 58
        #(10.1, 57.1),  # Atrium SW   Node 59
        #(6.3, 57.1),  # Atrium SSW   Node 60
        #(6.3, 50.7),  # South main doors   Node 61
        #(6.3, 44.8)]  # Atrium SE   Node 62




## Use the following type for BFS/DFS
##olin = ListGraph(63, nodeLocs)
## Or use this one for Best-First/A*
#olin = MapGraph(63, nodeLocs)


## 0) Home lab:  Lab hall
#olin.addEdge(0, 1)

## 1) Lab hall:  Home lab, East T, East L
#olin.addEdge(1, 2)
#olin.addEdge(1, 3)

## 2) East T:  Lab hall, 259 hall, 258 lab hall
#olin.addEdge(2, 4)
#olin.addEdge(2, 6)

## 3) East L:  Lab hall, 247 hall
#olin.addEdge(3, 19)

## 4) 259 lab hall:  East T, Right Angle door
#olin.addEdge(4, 5)

## 5) Right Angle door: 259 lab hall, Right angle center, 232-233 hall
#olin.addEdge(5, 30)
#olin.addEdge(5, 32)

## 6) 258 lab hall: East T, 258 lab, 256 lab hall,
#olin.addEdge(6, 7)
#olin.addEdge(6, 8)

## 7) 258 lab:  258 lab hall, 256 lab
#olin.addEdge(7, 9)

## 8) 256 lab hall: 258 lab hall, 256 lab, 250 center hall
#olin.addEdge(8, 9)
#olin.addEdge(8, 10)

## 9) 256 lab:  258 lab, 256 lab hall, Reading room
#olin.addEdge(9, 12)

## 10) 250 center hall:  256 lab hall, Reading room hall
##                       250 center
#olin.addEdge(10, 11)
#olin.addEdge(10, 15)

## 11) Reading room hall: 250 center hall, Reading room, 
##                        205 center hall
#olin.addEdge(11, 12)
#olin.addEdge(11, 13)

## 12) Reading room:  256 lab, Reading room hall

## 13) 205 center hall:  Reading room hall,
##                       Atrium center hall, 205 east
#olin.addEdge(13, 14)
#olin.addEdge(13, 17)

## 14) Atrium center hall:  Reading room hall, 205 center hall, Atrium CE
#olin.addEdge(14, 57)

## 15) 250 center: 250 center hall, 250 north
#olin.addEdge(15, 16)

## 16)  250 north:  250 center, 
#olin.addEdge(16, 23)

## 17) 205 east:  205 center hall, 205 north hall
#olin.addEdge(17, 18)
#olin.addEdge(17, 26)

## 18) 205 west: 205 east, 205 west atrium
#olin.addEdge(18, 54)

## 19) 247 hall: East L, 247 room, 245 hall
#olin.addEdge(19, 20)
#olin.addEdge(19, 21)

## 20) 247 room:  247 hall

## 21) 245 hall: 247 hall, 245 room, 250 north hall
#olin.addEdge(21, 22)
#olin.addEdge(21, 23)

## 22) 245 room:  245 hall

## 23) 250 north hall: 250 north, 245 hall, 243 hall 
#olin.addEdge(23, 24)

## 24) 243 hall:  250 north hall, 243 room, 205 north hall
#olin.addEdge(24, 25)
#olin.addEdge(24, 26)

## 25) 243 room:  243 hall

## 26) 205 north hall:  205 east, 243 hall, 241 hall
#olin.addEdge(26, 27)

## 27) 241 hall: 205 north hall, 241 room, Atrium north hall
#olin.addEdge(27, 28)
#olin.addEdge(27, 29)

## 28) 241 room: 241 hall

## 29) Atrium north hall: 241 hall, Atrium NE
#olin.addEdge(29, 50)

## 30) Right angle center: Right angle door, 232-233 hall, 230-231 hall
#olin.addEdge(30, 32)
#olin.addEdge(30, 35)

## 31) 233 room: 232-233 hall
#olin.addEdge(31, 32)

## 32) 232-233 hall: Right angle door, Right angle center, 233 room, 232 room, 
##                   230-231 hall
#olin.addEdge(32, 33)
#olin.addEdge(32, 35)

## 33) 232 room: 232-233 hall

## 34) 231 room: 230-231 hall
#olin.addEdge(34, 35)

## 35) 230-231 hall: Right angle center, 232-233 hall, 231 room, 230 room
##                   228-229 hall
#olin.addEdge(35, 36)
#olin.addEdge(35, 38)

## 36) 230 room: 230-231 hall

## 37) 229 room:  228-229 hall
#olin.addEdge(37, 38)

## 38) 228-229 hall: 230-231 hall, 229 room, 228 room, 226-227 hall
#olin.addEdge(38, 39)
#olin.addEdge(38, 41)

## 39) 228 room:  228-229 hall

## 40) 227 room: 226-227 hall
#olin.addEdge(40, 41)

## 41) 226-227 hall:  228-229 hall, 227 room, 226 room, 224-225 hall
#olin.addEdge(41, 42)
#olin.addEdge(41, 44)

## 42) 226 room:  226-227 hall

## 43) 225 room: 224-225 hall
#olin.addEdge(43, 44)

## 44) 224-225 hall: 226-227 hall, 225 room, 224 room, 222-223 hall
#olin.addEdge(44, 45)
#olin.addEdge(44, 47)

## 45) 224 room: 224-225 hall

## 46) 223 room: 222-223 hall
#olin.addEdge(46, 47)

## 47) 222-223 hall: 224-225 hall, 223 room, 222 room, Atrium south hall
#olin.addEdge(47, 48)
#olin.addEdge(47, 49)

## 48) 222 room: 222-223 hall

## 49) Atrium south hall: 222-223 hall, Atrium SE
#olin.addEdge(49, 62)

## 50) Atrium NE: Atrium north hall, 205 west atrium, North main doors, Atrium NC
#olin.addEdge(50, 54)
#olin.addEdge(50, 51)
#olin.addEdge(50, 53)
#olin.addEdge(50, 57)

## 51) North main doors: Atrium NE, Atrium NW, Atrium NC
#olin.addEdge(51, 52)
#olin.addEdge(51, 53)

## 52) Atrium NW:  North main doors, Atrium NC, Atrium CW
#olin.addEdge(52, 53)
#olin.addEdge(52, 55)

## 53) Atrium NC: Atrium NE, North main doors, Atrium NW, 205 west atrium
##                Atrium CW, Atrium CE
#olin.addEdge(53, 54)
#olin.addEdge(53, 55)
#olin.addEdge(53, 57)

## 54) 205 west atrium: 205 west, Atrium NE, Atrium NC, Atrium CE
#olin.addEdge(54, 57)

## 55) Atrium CW:  Atrium NW, Atrium NC, Atrium CC, Atrium SW, Atrium SC
#olin.addEdge(55, 56)
#olin.addEdge(55, 59)
#olin.addEdge(55, 58)

## 56) Atrium CC: Atrium CW, Atrium CE, Atrium SC
#olin.addEdge(56, 57)
#olin.addEdge(56, 58)

## 57) Atrium CE: Atrium NC, 205 west atrium, Atrium CC, Atrium SC, Atrium SE
##                Atrium NE
#olin.addEdge(57, 58)
#olin.addEdge(57, 62)

## 58) Atrium SC: Atrium CW, Atrium CC, Atrium CE, Atrium SW, Atrium SSW,
##                South main doors, Atrium SE
#olin.addEdge(58, 59)
#olin.addEdge(58, 60)
#olin.addEdge(58, 61)
#olin.addEdge(58, 62)

## 59) Atrium SW: Atrium CW, Atrium SC, Atrium SSW, South main doors
#olin.addEdge(59, 60)
#olin.addEdge(59, 61)

## 60)  Atrium SSW: Atrium SC, Atrium SW, South main doors
#olin.addEdge(60, 61)

## 61) South main doors: Atrium SC, Atrium SW, Atrium SSW, Atrium SE
#olin.addEdge(61, 62)

## 62) Atrium SE: Atrium south hall, Atrium CE, Atrium SC, South main doors





















