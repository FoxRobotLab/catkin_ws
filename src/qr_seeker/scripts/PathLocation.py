""" -------------------------------------------------------------------
File: PathLocation.py
Date: June 2016


# --------------------------------------------------------------------- """

import OlinGraph

class PathLocation(object):
    def __init__(self):
        self.destination = None
        self.pathTraveled = None


    def beginJourney(self):
        totalNumNodes = OlinGraph.olin._numVerts
        while self.destination is None:
            userInput = int(input("Enter destination index: "))
            if userInput in range(totalNumNodes):
                self.destination = userInput
        self.pathTraveled =[]


    def continueJourney(self, qrInfo):
        """Read things"""
        nodeNum, nodeCoord, nodeName = qrInfo
        heading = OlinGraph.olin.getMarkerInfo(nodeNum)

        if heading is None:
            heading = 0
        print("Location is ", nodeName, "with number", nodeNum, "at coordinates", nodeCoord)
        self.pathTraveled.append(nodeNum)
        print ("Path travelled so far: \n", self.pathTraveled)
        if nodeNum == self.destination:
            print ("Arrived at destination.")
            return None

        path = OlinGraph.olin.getShortestPath(nodeNum, self.destination)
        currentNode, nextNode = path[0], path[1]
        targetAngle = OlinGraph.olin.getAngle(currentNode, nextNode)

        print "Turning from node " , str(currentNode) , " to node " , str(nextNode)

        return heading, targetAngle


    def getPath(self):
        print("I'm serious. You actually did it. Here is your path again so you can see how far you have come.")
        return self.pathTraveled
