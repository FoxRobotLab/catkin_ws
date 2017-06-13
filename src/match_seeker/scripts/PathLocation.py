""" -------------------------------------------------------------------
File: PathLocation.py
Date: June 2016

This file deals with all path locations. It knows the current path traveled
and handles adding nodes to the path. Once the node is taken in PathLocation.py
asks OlinGraph about angles in order to send back how far the robot should
turn to get to the next node in its calculated path.
# --------------------------------------------------------------------- """

# import MapGraph
# from DataPaths import basePath


class PathLocation(object):
    """Handles the path planning component of the robot, interacting with the Olin MapGraph"""
    def __init__(self, mapPath, logWriter):
        self.olin = mapPath
        self.logger = logWriter
        self.destination = None
        self.pathTraveled = None


    def beginJourney(self):
        """Sets up the destination for the route."""
        self.destination = None
        totalNumNodes = self.olin.getSize()
        while self.destination is None:
            userInput = int(input("Enter destination index (99 to quit): "))
            if userInput == 99:
                return False
            if userInput in range(totalNumNodes):
                self.destination = userInput
        self.pathTraveled =[]
        return True


    def continueJourney(self, matchInfo):
        """This is given information about the marker that is currently seen. It adds it to the path traveled,
        and then it gets the best path from this point to the destination. This is used to determine the next
        direction and target node for the robot. Computing the shortest path is done by the MapGraph itself, as needed."""
        nodeNum, currHead = matchInfo

        self.logger.log("Location is number " + str(nodeNum))
        self.pathTraveled.append(nodeNum)
        self.logger.log("Path traveled so far:")
        self.logger.log(str(self.pathTraveled))
        if nodeNum == self.destination:
            self.logger.log("ARRIVED AT GOAL!!")
            return None

        self.goalPath = self.olin.getShortestPath(nodeNum, self.destination)
        currentNode, nextNode = path[0], path[1]
        targetAngle = self.olin.getAngle(currentNode, nextNode)

        self.logger.log("Turning from node " + str(currentNode) + " to node " + str(nextNode))

        return targetAngle


    def getCurrentPath(self):
        """Returns the path it wants to follow currently."""
        return self.goalPath


    def getPathTraveled(self):
        """Returns the current path traveled."""
        return self.pathTraveled
