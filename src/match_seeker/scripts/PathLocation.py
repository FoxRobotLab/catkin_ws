""" -------------------------------------------------------------------
File: PathLocation.py
Date: June 2016

This file deals with all path locations. It knows the current path traveled
and handles adding nodes to the path. Once the node is taken in PathLocation.py
asks OlinGraph about angles in order to send back how far the robot should
turn to get to the next node in its calculated path.
# --------------------------------------------------------------------- """

import MapGraph
from OSPathDefine import basePath


class PathLocation(object):
    """Handles the path planning component of the robot, interacting with the Olin MapGraph"""
    def __init__(self):
        self.destination = None
        self.pathTraveled = None
        self.path = basePath + "scripts/olinGraph.txt"
        self.olin = MapGraph.readMapFile(self.path)


    def beginJourney(self):
        """Sets up the destination for the route."""
        totalNumNodes = self.olin.getSize()
        while self.destination is None:
            userInput = int(input("Enter destination index: "))
            if userInput in range(totalNumNodes):
                self.destination = userInput
        self.pathTraveled =[]


    def continueJourney(self, matchInfo):
        """This is given information about the marker that is currently seen. It adds it to the path traveled,
        and then it gets the best path from this point to the destination. This is used to determine the next
        direction and target node for the robot. Computing the shortest path is done by the MapGraph itself, as needed."""
        nodeNum, nodeCoord, currHead = matchInfo
        # heading = self.olin.getMarkerInfo(nodeNum)

        # if heading is None:
        #     print "Heading for node", nodeNum, "is not specified, default heading = 0"
        #     heading = 0
        print("Location is number", nodeNum, "at coordinates", nodeCoord)
        self.pathTraveled.append(nodeNum)
        print ("Path travelled so far: \n", self.pathTraveled)
        if nodeNum == self.destination:
            print ("Arrived at destination.")
            return None

        path = self.olin.getShortestPath(nodeNum, self.destination)
        currentNode, nextNode = path[0], path[1]
        targetAngle = self.olin.getAngle(currentNode, nextNode)

        print("Turning from node ", str(currentNode), " to node ", str(nextNode))

        return targetAngle


    def getPath(self):
        """Returns the current path traveled."""
        # print("I'm serious. You actually did it. Here is your path again so you can see how far you have come.")
        return self.pathTraveled
